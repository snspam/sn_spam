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
import org.linqs.psl.model.atom.RandomVariableAtom
import org.linqs.psl.model.term.ConstantType
import org.linqs.psl.model.predicate.Predicate
// weight learning
import org.linqs.psl.application.learning.weight.em.HardEM
// inference
import org.linqs.psl.application.inference.MPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult

/**
 * Interpretability relational model object.
 *
 * Performs inference on the same network a number of time with varying
 * inital values and saves the prediction for each run.
 *
 * @author Jonathan Brophy
 */
public class Lime {
    private static final String W_PT = "write_pt"
    private static final String R_PT = "read_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m

    /**
     * Constructor.
     *
     * @param data_f folder to store temporary datastore in.
     */
    public Lime(String data_f) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = data_f + 'db/psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        print('\nData store setup at: ' + db_path)
    }

    /**
     * Specify and add predicate definitions to the model.
     */
    private void define_predicates() {
        ConstantType unique_id = ConstantType.UniqueID
        m.add predicate: "spam", types: [unique_id]
        m.add predicate: "indpred", types: [unique_id]
        m.add predicate: "intext", types: [unique_id, unique_id]
        m.add predicate: "posts", types: [unique_id, unique_id]
        m.add predicate: "intrack", types: [unique_id, unique_id]
        m.add predicate: "inhash", types: [unique_id, unique_id]
        m.add predicate: "inment", types: [unique_id, unique_id]
        m.add predicate: "invideo", types: [unique_id, unique_id]
        m.add predicate: "inhour", types: [unique_id, unique_id]
        m.add predicate: "inlink", types: [unique_id, unique_id]
        m.add predicate: "inhotel", types: [unique_id, unique_id]
        m.add predicate: "inrest", types: [unique_id, unique_id]
        m.add predicate: "spammytext", types: [unique_id]
        m.add predicate: "spammyuser", types: [unique_id]
        m.add predicate: "spammytrack", types: [unique_id]
        m.add predicate: "spammyhash", types: [unique_id]
        m.add predicate: "spammyment", types: [unique_id]
        m.add predicate: "spammyvideo", types: [unique_id]
        m.add predicate: "spammyhour", types: [unique_id]
        m.add predicate: "spammylink", types: [unique_id]
        m.add predicate: "spammyhotel", types: [unique_id]
        m.add predicate: "spammyrest", types: [unique_id]
    }

    /**
     * Load model rules from a text file.
     *
     *@param filename name of the text file with the model rules.
     */
    private void define_rules(String filename) {
        print('\nloading model...')
        m.addRules(new FileReader(filename))
    }

    /**
     * Parses a file containing each comment in the test set and their
     * altered starting values.
     *
     *@param filename name of the peturbed file.
     *@return a tuple containing a list of comment ids, and a list of lists
     * of various starting values for each comment.
     */
    private Tuple parse_perturbed_file(String filename) {
        def dataframe = []
        def com_ids_list = []
        def predictions_list = []
        def com_id_column = 0

        def file = new File(filename)
        if (file.exists()) {
            def lines = file.readLines()
            def num_columns = lines[0].tokenize(',').size()

            // initialize list of lists.
            for (int i = 0; i < num_columns; i++) {
                dataframe.add([])
            }

            // fill 2d array with values.
            for (int i = 1; i < lines.size(); i++) {  // skip header
                def columns = lines[i].tokenize(',')
                for (int j = 0; j < columns.size(); j++) {
                    dataframe[j].add(columns[j])
                }
            }
        }
        return new Tuple(dataframe[0], dataframe[1..-1])
    }

    /**
     * Open a csv file to write the generated labels to.
     *
     *@param fold experiment identifier.
     *@param out_f folder to write the labels to.
     *@return a FileWriter object.
     */
    private FileWriter open_labels_writer(int fold, String out_f) {
        FileWriter fw = new FileWriter(out_f + 'labels_' + fold + '.csv')
        fw.write('sample,com_id,pred\n')
        return fw
    }

    /**
     * Load validation and training predicate data.
     *
     *@param fold experiment identifier.
     *@param data_f folder to load data from.
     */
    private void load_data(int fold, String data_f) {
        print('\nloading data...')
        long start = System.currentTimeMillis()

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)

        // load test set comments to be labeled.
        load_file(data_f + 'test_no_label_' + fold, spam, write_pt)
        load_file(data_f + 'test_pred_' + fold, indpred, read_pt)

        // load relational data.
        load_file(data_f + 'test_intext_' + fold, intext, read_pt)
        load_file(data_f + 'test_text_' + fold, spammytext, write_pt)

        load_file(data_f + 'test_posts_' + fold, posts, read_pt)
        load_file(data_f + 'test_user_' + fold, spammyuser, write_pt)

        load_file(data_f + 'test_intrack_' + fold, intrack, read_pt)
        load_file(data_f + 'test_track_' + fold, spammytrack, write_pt)

        load_file(data_f + 'test_inhash_' + fold, inhash, read_pt)
        load_file(data_f + 'test_hash_' + fold, spammyhash, write_pt)

        load_file(data_f + 'test_inment_' + fold, inment, read_pt)
        load_file(data_f + 'test_ment_' + fold, spammyment, write_pt)

        load_file(data_f + 'test_invideo_' + fold, invideo, read_pt)
        load_file(data_f + 'test_video_' + fold, spammyvideo, write_pt)

        load_file(data_f + 'test_inhour_' + fold, inhour, read_pt)
        load_file(data_f + 'test_hour_' + fold, spammyhour, write_pt)

        load_file(data_f + 'test_inlink_' + fold, inlink, read_pt)
        load_file(data_f + 'test_link_' + fold, spammylink, write_pt)

        load_file(data_f + 'test_inhotel_' + fold, inhotel, read_pt)
        load_file(data_f + 'test_hotel_' + fold, spammyhotel, write_pt)

        load_file(data_f + 'test_inrest_' + fold, inrest, read_pt)
        load_file(data_f + 'test_rest_' + fold, spammyrest, write_pt)

        long end = System.currentTimeMillis()
        print(((end - start) / 1000.0) + 's')   
    }

    /**
     * Loads a tab separated predicate data file. Automatically handles
     * truth and non truth files.
     *
     *@param filename name of the file to load.
     *@param predicate name of the predicate to load data for.
     *@param partition parition to load the file into.
     */
    private void load_file(filename, predicate, partition) {
        String file = filename + '.tsv'
        if (new File(file).exists()) {
            Inserter inserter = this.ds.getInserter(predicate, partition)
            InserterUtils.loadDelimitedDataAutomatic(predicate, inserter, file)
        }
    }

    /**
     * Specifies which predicates are closed (i.e. observations that cannot
     * be changed).
     *
     *@return a set of closed predicates.
     */
    private Set<Predicate> define_closed_predicates() {
        Set<Predicate> closed = [indPred, inText, posts, inTrack,
                inHash, inMent, inVideo, inHour]
        return closed
    }

    /**
     * Prints the number of samples that have been processed so far.
     *
     *@param j sample number.
     */
    private void print_progress(int j) {
        if (j % 25 == 0) {
            print('\nFinished sample ' + j + '...')
        }
    }

    /**
     * Runs inference.
     *
     *@param set of closed predicates.
     *@return a result object from inference.
     */
    private FullInferenceResult run_inference(Set<Predicate> closed) {
        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)

        Database inference_db = this.ds.getDatabase(write_pt, closed, read_pt)
        MPEInference mpe = new MPEInference(this.m, inference_db, this.cb)
        FullInferenceResult result = mpe.mpeInference()
        mpe.close()
        mpe.finalize()
        inference_db.close()

        return result
    }

    /**
     * Update indpred atoms with new values.
     *
     *@param com_ids list of comment ids.
     *@param preds list of starting values for each comment id.
     */
    private void update_perturbed_instance(def com_ids, def preds) {
        Partition read_pt = this.ds.getPartition(R_PT)
        Database read_db = this.ds.getDatabase(read_pt)

        // build dict of com_ids and predictions.
        def m = [:]
        for (int i = 0; i < com_ids.size(); i++) {
            m[com_ids[i]] = preds[i]
        }

        // update indpred atoms.
        for (RandomVariableAtom atom : Queries.getAllAtoms(read_db, IndPred)) {
           String com_id = atom.getArguments()[0].toString().replace("'", "")
           float val = Float.parseFloat(m[com_id])
           atom.setValue(val)
           atom.commitToDB()
        }
        read_db.close();
    }

    /**
     * Writes the prediction for the target comment id after inference.
     *
     *@param j sample number.
     *@param target_id id of the comment needing an explanation.
     *@param fw handle to the writing object.
     */
    private void write_prediction(int j, String target_id, FileWriter fw) {
        Partition write_pt = this.ds.getPartition(W_PT)
        Database predictions_db = this.ds.getDatabase(write_pt)

        DecimalFormat formatter = new DecimalFormat("#.#####")

        for (GroundAtom atom : Queries.getAllAtoms(predictions_db, spam)) {
            String com_id = atom.getArguments()[0].toString().replace("'", "")
            if (com_id == target_id) {
                def pred = formatter.format(atom.getValue())
                fw.write('n_' + j + ',' + com_id + ',' + pred + '\n')
            }
        }
        predictions_db.close()
    }

    /**
     * Specifies the model, and runs inference for the given number
     * of perturbed samples, and saves the predictions from those runs.
     *
     *@param target_id id of the comment needing an explanation.
     *@param fold experiment id.
     *@param relations list of relations present in the data.
     *@param model_f folder to read the model from.
     *@param data_f folder to write and read predicate data to/from.
     *@param out_f folder to output predictions to.
     */
    private void run(String target_id, int fold, def relations,
            String model_f,  String data_f, String out_f) {
        String rules_filename = model_f + 'rules_' + fold + '.txt'
        String perturbed_filename = data_f + 'perturbed.csv'

        define_predicates()
        define_rules(rules_filename)
        load_data(fold, data_f)
        Set<Predicate> closed = define_closed_predicates()

        def (com_ids, samples) = parse_perturbed_file(perturbed_filename)
        FileWriter fw = open_labels_writer(fold, out_f)
        for (int j = 0; j < samples.size(); j++) {
            print_progress(j)
            update_perturbed_instance(com_ids, samples[j])
            run_inference(closed)
            write_prediction(j, target_id, fw)
        }
        fw.close()
        this.ds.close()
    }

    /**
     * Specifies relative paths to 'psl' directory.
     *
     *@param domain social network (e.g. soundcloud, youtube, twitter, etc.).
     *@return a tuple with various data folder paths.
     */
    public static Tuple define_file_folders(String domain) {
        String model_f = '../output/' + domain + '/models/'
        String data_f = './data/' + domain + '/interpretability/'
        String out_f = '../output/' + domain + '/interpretability/'
        new File(out_f).mkdirs()
        return new Tuple(model_f, data_f, out_f)
    }

    /**
     * Check and parse commandline arguments.
     *
     *@param args arguments from the commandline.
     *@return a tuple containing the target comment id, experiment id,
     * social network, and a list of relations present in the data.
     */
    public static Tuple check_commandline_args(String[] args) {
        if (args.length < 3) {
            print('Example: [target_id] [fold] [domain] [relations]')
            System.exit(0)
        }
        String target_id = args[0].toString();
        int fold = args[1].toInteger();
        String domain = args[2].toString();
        def relations = []
        for (int i = 3; i < args.length; i++) {
            relations.add(args[i].toString());
        }
        return new Tuple(target_id, fold, domain, relations)
    }

    /**
     * Main method that creates and runs the Interpretability object.
     *
     *@param args arguments from the commandline.
     */
    public static void main(String[] args) {
        def (target_id, fold, domain, relations) = check_commandline_args(args)
        def (model_f, data_f, out_f) = define_file_folders(domain)
        Lime lime = new Lime(data_f)
        lime.run(target_id, fold, relations, model_f, data_f, out_f)
    }
}
