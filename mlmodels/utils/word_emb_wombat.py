from wombat_api.core import connector as wb_conn

StrFeedbackErrorCode = '*PAD*'
Wombat_wec_ids = "algo:glove;dataset:42b;dims:300;fold:1;unit:token;norm:none"


class Wombat(object):
    def __init__(self, wombat_path=str()):
        self.wombat_path = wombat_path
        try:
            print('Connecting to wombat path:', self.wombat_path)
            self.wbc = wb_conn(path=self.wombat_path, create_if_missing=False)
        except:
            self.wbc = None
            print("ERROR: unable to locate {} file".format(self.wombat_path))

    def get(self, word, default_embedding=None):
        if self.wbc is None:
            print("ERROR: unable to locate {} file".format(self.wombat_path))
            return default_embedding
        else:
            embedding_object = self.wbc.get_vectors(Wombat_wec_ids, {}, for_input=[[word]], in_order=False)
            embedding_str_feedback = embedding_object[0][1][0][2][0][0]
            embedding_ndarray = embedding_object[0][1][0][2][0][1]
            if embedding_str_feedback == StrFeedbackErrorCode:
                print('ERROR: NO EMBEDDING {} FOUND IN Wombat'.format(word))
                return default_embedding
            else:
                return embedding_ndarray

    def get_vectors(self, inputs):
        return self.wbc.get_vectors(Wombat_wec_ids, {}, for_input=inputs, in_order=False)


if __name__ == '__main__':
    wombat_path = "/Users/media/data/embeddings/database/glove-sqlite_"
    wombat_object = Wombat(wombat_path)
    pass
