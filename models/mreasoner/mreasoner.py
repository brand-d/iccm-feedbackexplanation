import ccobra
import numpy as np

class MReasoner(ccobra.CCobraModel):
    def __init__(self, name='mReasoner'):
        super(MReasoner, self).__init__(name, ['syllogistic'], ['single-choice'])

        # Prepare cache
        self.cache = np.load('cache/2020-09-09-cache-11-10.npy')
        self.n_epsilon, self.n_lambda, self.n_omega, self.n_sigma = self.cache.shape[:-2]
        self.params = None

        # Normalize cache
        self.cache /= self.cache.sum(-1, keepdims=True)

    def end_participant(self, identifier, model_log, **kwargs):
        paramnames = ['epsilon', 'lambda', 'omega', 'sigma']
        for pname, (_, pval) in zip(paramnames, self.params):
            model_log[pname] = pval

    def pre_train(self, dataset, **kwargs):
        tdata = np.zeros((64, 9))
        for subj_data in dataset:
            for task_data in subj_data:
                syl = ccobra.syllogistic.Syllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resp = syl.encode_response(task_data['response'])

                idx_task = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
                idx_resp = ccobra.syllogistic.RESPONSES.index(enc_resp)
                tdata[idx_task, idx_resp] += 1

        # Perform fitting
        self.fit(tdata)

    def pre_train_person(self, data, **kwargs):
        self.pre_train([data])

    def fit(self, tdata):
        best_score = -1
        best_params = None

        # Iterate over parameterizations in the cache
        for idx_epsilon, p_epsilon in enumerate(np.linspace(0, 1, self.n_epsilon)):
            for idx_lambda, p_lambda in enumerate(np.linspace(0.1, 8, self.n_lambda)):
                for idx_omega, p_omega in enumerate(np.linspace(0, 1, self.n_omega)):
                    for idx_sigma, p_sigma in enumerate(np.linspace(0, 1, self.n_sigma)):
                        params = (idx_epsilon, idx_lambda, idx_omega, idx_sigma)
                        cache_mat = self.cache[params]

                        # Compare cache with training data
                        score = np.sum(cache_mat * tdata)

                        if score > best_score:
                            best_score = score
                            best_params = list(zip(params, (p_epsilon, p_lambda, p_omega, p_sigma)))

        # Set to best params
        self.params = best_params

    def predict(self, item, **kwargs):
        # Obtain task information
        syl = ccobra.syllogistic.Syllogism(item)
        enc_task = syl.encoded_task
        idx_task = ccobra.syllogistic.SYLLOGISMS.index(enc_task)

        # Obtain prediction matrix
        param_idxs = tuple(x[0] for x in self.params)
        cache_mat = self.cache[param_idxs]
        cache_pred = cache_mat[idx_task]

        # Generate prediction
        pred_idxs = np.arange(len(cache_pred))[cache_pred == cache_pred.max()]
        pred = ccobra.syllogistic.RESPONSES[np.random.choice(pred_idxs)]
        return syl.decode_response(pred)
