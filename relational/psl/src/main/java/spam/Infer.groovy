package spam

// Java imports
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
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
// inference
import org.linqs.psl.application.inference.LazyMPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult
// evaluation
import org.linqs.psl.utils.evaluation.statistics.RankingScore
import org.linqs.psl.utils.evaluation.statistics.SimpleRankingComparator

/**
 * Infer relational model object.
 *
 * Defines all aspects of the model, loads the data, learns weights,
 * and runs inference.
 *
 * @author Jonathan Brophy
 */
public class Infer {
    private static final String W_PT = "write_pt"
    private static final String R_PT = "read_pt"
    private static final String L_PT = "labels_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m
    private PrintWriter fw

    /**
     * Constructor.
     *
     * @param data_f folder to store temporary datastore in.
     */
    public Infer(String data_f, status_f, fold) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = data_f + 'db/psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        // this.fw = new PrintWriter(System.out)
    }

    private void out(String message, def newline=1) {
        String msg = newline == 1 ? '\n' + message : message
        // this.fw.print(msg)
        // this.fw.flush()
    }

    private void time(long t1, def suffix='m') {
        long elapsed = System.currentTimeMillis() - t1

        if (suffix == 's') {
            elapsed /= 1000.0
        }
        else if (suffix == 'm') {
            elapsed /= (1000.0 * 60.0)
        }
        else if (suffix == 'h') {
            elapsed /= (1000.0 * 60.0 * 60)
        }
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
     * Load validation and training predicate data.
     *
     *@param fold experiment identifier.
     *@param data_f folder to load data from.
     *@closed list of closed predicate names.
     */
    private void load_data(int fold, String data_f, def closed) {
        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)
        Partition labels_pt = this.ds.getPartition(L_PT)

        def pre = 'test_'

        // load test set comments to be labeled.
        load_file(data_f + pre + fold, 'spam', labels_pt)
        load_file(data_f + pre + 'no_label_' + fold, 'spam', write_pt)
        load_file(data_f + pre + 'pred_' + fold, 'indPred', read_pt)

        // load relational data.
        for (def pred: closed) {
            def relation = pred
            def group = pred.replace('has', '')
            def rel_fname = data_f + pre + relation + '_' + fold
            def group_fname = data_f + pre + group + '_' + fold

            load_file(rel_fname, relation, read_pt)
            load_file(group_fname, 'spmy' + group, write_pt)
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
     * Run inference with the trained model on the test set.
     *
     *@param set of closed predicates.
     *@return a FullInferenceResult object.
     */
    private FullInferenceResult run_inference(closed_preds) {
        long start = System.currentTimeMillis()

        Set<Predicate> closed = closed_preds.collect{this.m.getPredicate(it)}

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)

        Database inference_db = this.ds.getDatabase(write_pt, closed, read_pt)
        LazyMPEInference mpe = new LazyMPEInference(this.m, inference_db,
                this.cb)
        FullInferenceResult result = mpe.mpeInference()
        mpe.close()
        mpe.finalize()
        inference_db.close()

        time(start)
        return result
    }

    private void evaluate(Set<Predicate> closed) {
        long start = System.currentTimeMillis()

        Partition labels_pt = this.ds.getPartition(L_PT)
        Partition write_pt = this.ds.getPartition(W_PT)
        Partition temp_pt = this.ds.getPartition('evaluation_pt')

        Database labels_db = this.ds.getDatabase(labels_pt, closed)
        Database predictions_db = this.ds.getDatabase(temp_pt, write_pt)

        def comparator = new SimpleRankingComparator(predictions_db)
        comparator.setBaseline(labels_db)

        def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,
                RankingScore.AreaROC]
        double[] score = new double[metrics.size()]

        for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(spam)
        }

        time(start)

        // out('AUPR: ' + score[0].trunc(4))
        // out(', N-AUPR: ' + score[1].trunc(4), 0)
        // out(', AUROC: ' + score[2].trunc(4), 0)

        labels_db.close()
        predictions_db.close()
    }

    /**
     * Print inference result information.
     *
     *@param r object resulting from inference.
     */
    private void print_inference_info(FullInferenceResult r) {
        float incomp = r.getTotalWeightedIncompatibility().trunc(2)
        int grnd_atoms = r.getNumGroundAtoms()
        int grnd_evd = r.getNumGroundEvidence()
        def s = 'incompatibility: ' + incomp.toString()
        s += ', ground atoms: ' + grnd_atoms.toString()
        s += ', ground evidence: ' + grnd_evd.toString()
    }

    /**
     * Write the relational model predictions for each comment in the test set.
     *
     *@param fold experiment identifier.
     *@param pred_f folder to save predictions to.
     */
    private void write_predictions(int fold, String pred_f) {
        long start = System.currentTimeMillis()

        Partition temp_pt = this.ds.getPartition('temp_pt')
        Partition write_pt = this.ds.getPartition(W_PT)
        Database predictions_db = this.ds.getDatabase(temp_pt, write_pt)

        DecimalFormat formatter = new DecimalFormat("#.#####")
        FileWriter fw = new FileWriter(pred_f + 'psl_preds_' + fold + '.csv')

        fw.write('com_id,psl_pred\n')
        for (GroundAtom atom : Queries.getAllAtoms(predictions_db, spam)) {
            double pred = atom.getValue()
            String com_id = atom.getArguments()[0].toString().replace("'", "")
            fw.write(com_id + ',' + formatter.format(pred) + '\n')
        }
        fw.close()
        predictions_db.close()

        time(start)
    }

    /**
     * Method to define the model, learn weights, and perform inference.
     *
     *@param fold experiment identifier.
     *@param iden identifier for subgraph to reason over.
     *@param data_f data folder.
     *@param pred_f predictions folder.
     *@param model_f model folder.
     */
    private void run(int fold, int iden, String data_f, String pred_f,
                     String model_f) {
        String rules_filename = model_f + 'rules_' + fold + '.txt'

        def (predicates, params, closed) = extract_predicates(rules_filename)
        define_predicates(predicates, params)
        define_rules(rules_filename)
        load_data(iden, data_f, closed)
        FullInferenceResult result = run_inference(closed)
        print_inference_info(result)
        write_predictions(iden, pred_f)

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
        String pred_f = '../output/' + domain + '/predictions/'
        String model_f = '../output/' + domain + '/models/'
        String status_f = '../output/' + domain + '/status/'
        new File(pred_f).mkdirs()
        new File(model_f).mkdirs()
        return new Tuple(data_f, pred_f, model_f, status_f)
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
     * Main method that creates and runs the Infer object.
     *
     *@param args commandline arguments.
     */
    public static void main(String[] args) {
        def (fold, iden, domain) = check_commandline_args(args)
        def (data_f, pred_f, model_f, status_f) = define_file_folders(domain)
        Infer b = new Infer(data_f, status_f, fold)
        b.run(fold, iden, data_f, pred_f, model_f)
    }
}
