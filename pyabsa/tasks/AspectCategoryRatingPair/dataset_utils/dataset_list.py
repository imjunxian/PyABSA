# dataset_utils/dataset_list.py
from pyabsa.utils.data_utils.dataset_item import DatasetItem

class T5DatasetList(list):
    GoogleReviews = DatasetItem("GoogleReviews", "GoogleReviewsData") 

    def __init__(self):
        super(T5DatasetList, self).__init__([self.GoogleReviews])