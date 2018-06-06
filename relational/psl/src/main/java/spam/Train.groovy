package spam

// Java imports
import java.io.File
import java.io.FileWriter
import java.text.DecimalFormat
// PSL imports
import org.linqs.psl.config.ConfigBundle
import org.linqs.psl.config.ConfigManager
// database
import org.linqs.psl.database.Partition
import org.linqs.psl.database.DataStore
import org.linqs.psl.database.Database
import org.linqs.psl.database.Queries
import org.linqs.psl.database.loading.Inserter
import org.linqs.psl.database.rdbms.RDBMSDataStore
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver.Type
// data loading
import org.linqs.psl.utils.dataloading.InserterUtils
// model
import org.linqs.psl.groovy.PSLModel
import org.linqs.psl.model.rule.Rule
import org.linqs.psl.model.atom.GroundAtom
import org.linqs.psl.model.term.ConstantType
import org.linqs.psl.model.predicate.Predicate
// weight learning
import org.linqs.psl.application.learning.weight.em.HardEM

/**
 * Train a relational model object.
 *
 * Defines all aspects of the model, loads the data, learns weights,
 * and runs inference.
 *
 * @author Jonathan Brophy
 */
public class Train {
    private static final String WL_W_PT = "wl_write_pt"
    private static final String WL_R_PT = "wl_read_pt"
    private static final String WL_L_PT = "wl_labels_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m
    private File fw

    /**
     * Constructor.
     *
     * @param data_f folder to store temporary datastore in.
     */
    public Train(String data_f, String status_f, int fold) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = data_f + 'db/psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        this.fw = new File(status_f + 'train_' + fold + '.txt')
        this.fw.append('\ndata store setup at: ' + db_path)
    }

    /**
     * Specify and add predicate definitions to the model.
     */
    private void define_predicates(predicates, params) {
        ConstantType unique_id = ConstantType.UniqueID
        def sgl = [unique_id]
        def dbl = [unique_id, unique_id]

        for (int i = 0; i < predicates.size(); i++) {
            def pred = predicates[i]
            def type = params[i] == 1 ? sgl : dbl
            this.m.add predicate: pred, types: type
        }
    }

    /**
     * Load model rules from a text file.
     *
     *@param filename name of the text file with the model rules.
     */
    private void define_rules(String filename) {
        this.m.addRules(new FileReader(filename))
    }

    /**
     * Extract predicates from the untrained model.
     *
     *@param filename name of the text file with the model rules.
     *@returns tuple of predicate names, list of parameters per predicate,
     *  and the names of the closed predicates.
     */
    private Tuple extract_predicates(String filename) {
        def regex = /\s([a-z]+)\(/
        def file = new File(filename)
        def lines = file.readLines()

        def predicates = []
        for (String line: lines) {
            def line_preds = line.findAll(regex).collect{it.replace('(', '')}
            line_preds = line_preds.collect{it.replace(' ', '')}
            predicates += line_preds
        }
        predicates = predicates.toSet().toList()

        def closed = predicates.findAll{!it.contains('spam')}
        def params = predicates.collect{!it.contains('spmy')\
            && (it.contains('has')) ? 2 : 1}
        return new Tuple(predicates, params, closed)
    }

    /**
     * Load validation and training predicate data.
     *
     *@param fold experiment identifier.
     *@param data_f folder to load data from.
     *@closed list of closed predicate names.
     */
    private void load_data(int fold, String data_f, def closed) {
        Partition wl_write_pt = this.ds.getPartition(WL_W_PT)
        Partition wl_read_pt = this.ds.getPartition(WL_R_PT)
        Partition wl_labels_pt = this.ds.getPartition(WL_L_PT)

        def pre = 'val_'

        // load test set comments to be labeled.
        load_file(data_f + pre + fold, 'spam', wl_labels_pt)
        load_file(data_f + pre + 'no_label_' + fold, 'spam', wl_write_pt)
        load_file(data_f + pre + 'pred_' + fold, 'indPred', wl_read_pt)

        // load relational data.
        for (def pred: closed) {
            def relation = pred
            def group = pred.replace('has', '')
            def rel_fname = data_f + pre + relation + '_' + fold
            def group_fname = data_f + pre + group + '_' + fold

            load_file(rel_fname, relation, wl_read_pt)
            load_file(group_fname, 'spmy' + group, wl_write_pt)
        }
    }

    /**
     * Loads a tab separated predicate data file. Automatically handles
     * truth and non truth files.
     *
     *@param filename name of the file to load.
     *@param predicate_name name of the predicate to load data for.
     *@param partition parition to load the file into.
     */
    private void load_file(filename, predicate_name, partition) {
        String file = filename + '.tsv'
        def predicate = this.m.getPredicate(predicate_name)

        if (new File(file).exists()) {
            Inserter inserter = this.ds.getInserter(predicate, partition)
            InserterUtils.loadDelimitedDataAutomatic(predicate, inserter, file)
        }
    }

    /**
     * Learn weights for model rules using vlidation data.
     *
     *@param closed set of closed predicates.
     */
    private void learn_weights(closed_preds) {
        Set<Predicate> closed = closed_preds.collect{this.m.getPredicate(it)}
        Set<Predicate> closed_labels = [spam]

        Partition wl_wr_pt = ds.getPartition(WL_W_PT)
        Partition wl_r_pt = ds.getPartition(WL_R_PT)
        Partition wl_l_pt = ds.getPartition(WL_L_PT)

        this.fw.append('\nlearning weights...')
        long start = System.currentTimeMillis()

        Database wl_tr_db = this.ds.getDatabase(wl_wr_pt, closed, wl_r_pt)
        Database wl_l_db = ds.getDatabase(wl_l_pt, closed_labels)

        HardEM w_learn = new HardEM(this.m, wl_tr_db, wl_l_db, this.cb)
        w_learn.learn()
        wl_tr_db.close()
        wl_l_db.close()

        long end = System.currentTimeMillis()
        this.fw.append(((end - start) / 60000.0) + 'm')
    }

    /**
     * Write the model with learned weights to a text file.
     *
     *@param fold experiment identifier.
     *@param model_f folder to save model to.
     */
    private void write_model(int fold, String model_f) {
        FileWriter mw = new FileWriter(model_f + 'rules_' + fold + '.txt')
        for (Rule rule : this.m.getRules()) {
            String rule_str = rule.toString().replace('~( ', '~').toLowerCase()
            String rule_filtered = rule_str.replace('( ', '').replace(' )', '')
            this.fw.append('\n\t' + rule_str)
            mw.write(rule_filtered + '\n')
        }
        this.fw.append('\n')
        mw.close()
    }

    /**
     * Method to define the model, learn weights, and perform inference.
     *
     *@param fold experiment identifier.
     *@param data_f data folder.
     *@param pred_f predictions folder.
     *@param model_f model folder.
     */
    private void run(int fold, String data_f, String model_f) {
        String rules_filename = data_f + 'rules_' + fold + '.txt'

        def (predicates, params, closed) = extract_predicates(rules_filename)
        define_predicates(predicates, params)
        define_rules(rules_filename)
        load_data(fold, data_f, closed)
        learn_weights(closed)
        write_model(fold, model_f)

        this.ds.close()
    }

    /**
     * Specifies relative paths to 'psl' directory.
     *
     *@param domain social network (e.g. soundcloud, youtube, twitter, etc.).
     *@return a tuple with various data folder paths.
     */
    public static Tuple define_file_folders(String domain) {
        String data_f = './data/' + domain + '/'
        String model_f = '../output/' + domain + '/models/'
        String status_f = '../output/' + domain + '/status/'
        new File(model_f).mkdirs()
        return new Tuple(data_f, model_f, status_f)
    }

    /**
     * Check and parse commandline arguments.
     *
     *@param args arguments from the commandline.
     *@return a tuple containing the experiment id and social network.
     */
    public static Tuple check_commandline_args(String[] args) {
        if (args.length < 3) {
            print('Missing args, example: [fold] [domain] [relations (opt)]')
            System.exit(0)
        }
        int fold = args[0].toInteger()
        int iden = args[1].toInteger()
        String domain = args[2].toString()
        return new Tuple(fold, iden, domain)
    }

    /**
     * Main method that creates and runs the Train object.
     *
     *@param args commandline arguments.
     */
    public static void main(String[] args) {
        def (fold, iden, domain) = check_commandline_args(args)
        def (data_f, model_f, status_f) = define_file_folders(domain)
        Train b = new Train(data_f, status_f, fold)
        b.run(fold, data_f, model_f)
    }
}
