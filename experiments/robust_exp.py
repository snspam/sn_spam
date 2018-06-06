"""
Module to test the effectiveness of a relational model against adversarial
manipulations.
"""


class Robust_Experiment:
    """Handles all operations to test spammers using multiple accounts."""

    def __init__(self, config_obj, runner_obj, modified=True, pseudo=True):
        """Initializes object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.runner_obj = runner_obj
        """Runs different parts of the application."""
        self.config_obj.modified = modified
        """Boolean to use relabeled data if True."""
        self.config_obj.pseudo = pseudo
        """Boolean to use all features if True."""

    # public
    def run_experiment(self):
        """Runs the independent and relational models, then changes all
                spammer ids to be unique and runs both models again."""
        self.single_run()
        self.change_config_parameters(alter_user_ids=True)
        self.single_run()

    # private
    def single_run(self):
        """Operations to run the independent model, train the relational model
                from those predictions, and then do joint prediction using
                the relational model on the test set."""
        val_df, test_df = self.runner_obj.run_independent()
        self.change_config_rel_op(train=True)
        self.runner_obj.run_relational(val_df, test_df)
        self.change_config_rel_op(train=False)
        self.runner_obj.run_relational(val_df, test_df)
        self.runner_obj.run_evaluation(test_df)

    def change_config_parameters(self, alter_user_ids=False):
        """Changes configuration options to alter spammer ids, and increases
                the experiment number."""
        self.config_obj.alter_user_ids = alter_user_ids
        self.config_obj.fold = str(int(self.config_obj.fold) + 1)
