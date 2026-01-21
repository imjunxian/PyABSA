# pyabsa/tasks/AspectCategoryRatingPair/trainer/trainer.py

from typing import Union
from pyabsa.framework.flag_class.flag_template import (
    DeviceTypeOption,
    ModelSaveOption,
)
from pyabsa.framework.trainer_class.trainer_template import Trainer
from ..configuration.configuration import T5ConfigManager
from ..prediction.predictor import T5Predictor
from ..instructor.instructor import T5TrainingInstructor
from ..models.model import T5GenModel 

class T5Trainer(Trainer):
    def __init__(
        self,
        config: T5ConfigManager = None,
        dataset=None,
        from_checkpoint: str = None,
        checkpoint_save_mode: int = ModelSaveOption.SAVE_MODEL_STATE_DICT,
        auto_device: Union[bool, str] = DeviceTypeOption.AUTO,
        path_to_save=None,
        load_aug=False,
    ):
        if not config:
            config = T5ConfigManager.get_t5_config_template()
            
        config.model = T5GenModel
        
        if not hasattr(config, "dataset_name") or not config.dataset_name:
            config.dataset_name = "GoogleReviews"

        # Initialize parent
        super(T5Trainer, self).__init__(
            config=config,
            dataset=dataset,
            from_checkpoint=from_checkpoint,
            checkpoint_save_mode=checkpoint_save_mode,
            auto_device=auto_device,
            path_to_save=path_to_save,
            load_aug=load_aug,
        )

        self.training_instructor = T5TrainingInstructor
        self.inference_model_class = T5Predictor
        
        self.config.task_code = "Generative_ABSA"
        self.config.task_name = "Aspect Sentiment Generation"

        # Start the custom run method
        self._run()

    def _run(self):
        """
        Override the parent _run method to bypass 'detect_dataset'.
        Since we set config.dataset_file manually in run_train.py, 
        we can directly start the Instructor.
        """
        # 1. Initialize Instructor
        self.instructor = self.training_instructor(self.config)
        
        # 2. Start Training
        return self.instructor.run()