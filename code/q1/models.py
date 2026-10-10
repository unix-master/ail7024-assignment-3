import numpy as np

class GMM:
    def __init__(self,n_components):
        self.n_components = n_components
        # in gmm, we can a single categorical hidden variable
        # self.z = np.eye((n_components))
        self.means = None
        self.covariance = None
        self.mixing_coefs = None

    def fit(self,X,max_steps=100):
        n_samples, n_features = X.shape
        # E step: Q(z) := gamma(z_ik) = P(z|x,theta)
        ## parameters
        self.means = np.zeros((self.n_components,n_features))
        self.covariance = np.array([np.eye(n_features) for _ in range(self.n_components)])
        #-- assign equal probability initially to each component
        self.mixing_coefs = (1/self.n_components)*np.ones(self.n_components)
        #
        self.gamma = np.empty((n_samples,self.n_components))
        ##
        # def gaussian_pdf(mu,sigma):
        #     sigma_det = np.linalg.det(sigma)
        #     sigma_inv = np.linalg.inv(sigma)
        #     # similarly power will overflow here for high-dimensional data
        #     denom = (2*np.pi*sigma_det)**(n_features/2)
        #     ##!! this will not work. inverting a huge matrix is unstable
        #     return np.exp(-0.5*(X-mu)@sigma_inv@(X-mu).T) / denom
        ##
        # norm_constant = np.sum([self.mixing_coefs[j]*gaussian_pdf(self.means[j],self.covariance[j]) \
        #                         for j in range(self.n_components)])
        n_steps = 0

        gaussian_pdfs = np.empty((n_samples,self.n_components))

        while n_steps <= max_steps:
            print(f'Iteration {n_steps+1}')
            prev_means = self.means.copy()
            for k in range(self.n_components):
                # sigma = LL^T
                L_k = np.linalg.cholesky(self.covariance[k])
                # assert False, L_k.shape
                log_det_k = 2*np.sum(np.log(np.diag(L_k)))
                # v = np.linalg.inv(L_k)@(X-self.means[k])
                # assert False, f'{self.means[k].shape}'
                v = np.linalg.solve(L_k,(X-self.means[k]).T).T
                # assert False, v.shape
                dist = np.sum(v**2,axis=1)
                log_pdf = -0.5*n_features*np.log(2*np.pi)-0.5*log_det_k-0.5*dist
                log_prior = np.log(self.mixing_coefs[k])
                gaussian_pdfs[:,k] = log_pdf + log_prior
            # log_norm_constant = np.sum(np.exp(gaussian_pdfs))
            max_vals = np.max(gaussian_pdfs,axis=1,keepdims=True)
            log_norm_constant = max_vals + np.log(np.sum(np.exp(gaussian_pdfs - max_vals),axis=1,keepdims=True))
            self.gamma = np.exp(gaussian_pdfs - log_norm_constant)
            # assert False, self.gamma.shape

            # M step: Q'(theta,theta_old) = max_{theta} Sum_{z} P(z|x,theta) * P(x,z|theta_old)
            for k in range(self.n_components):
                # self.gamma[:,k] = self.mixing_coefs[k]*gaussian_pdf(self.means[k],self.covariance[k])
                # self.gamma[:,k] /= norm_constant
                ## -----  
                gamma_k = np.sum(self.gamma[:,k],axis=0)
                mean_k_new = X.T@self.gamma[:,k] / gamma_k
                #!! covariance(t+1) should be calculated around the new mean at t+1
                cov_k_new = ((X-mean_k_new)*self.gamma[:,k].reshape(-1,1)).T@(X-mean_k_new) / gamma_k
                mixing_k_new = gamma_k / n_samples
                self.means[k] = mean_k_new
                self.covariance[k] = cov_k_new + 1e-6 * np.eye(*self.covariance[k].shape)
                self.mixing_coefs[k] = mixing_k_new
            n_steps += 1
            if np.max(np.abs(self.means-prev_means)) < 1e-6:
                break
        print(f'Algorithm stopped after {n_steps} steps.')

    def generate(self,z,size=1):
        assert z <= self.n_components
        return np.random.multivariate_normal(self.means[z],self.covariance[z],size=size)


class BMM:
    def __init__(self,n_components):
        self.n_components = n_components

    def fit(self,X,max_steps=10):
        n_samples, n_features = X.shape
        ## parameters
        self.mixing_coefs = np.ones(self.n_components) / self.n_components
        # -- probability of success in multivariate bernoulli
        # self.p = np.ones((self.n_components,n_features)) / n_features
        self.p = np.random.uniform(0.25,0.75,size=(self.n_components,n_features))
        ## ----
        self.gamma = np.empty((n_samples,self.n_components))
        n_steps = 0
        prev_log_likelihood = 0
        while n_steps <= max_steps:
            print(f'Iteration {n_steps}')
            # E step (compute gamma)
            log_prior = np.log(self.mixing_coefs)
            # assert False, (X@(np.log(self.p).T)).shape
            self.p = np.clip(self.p,1e-6,1-1e-6)
            log_pmfs = X@(np.log(self.p).T) + (1-X)@(np.log(1-self.p).T) # (n_samples,n_components)
            log_joint = log_prior + log_pmfs
            log_likelihood = np.sum(log_joint)
            max_vals = np.max(log_joint,axis=1,keepdims=True)
            # log_denom = max_vals + np.log(np.sum(log_pmfs*self.mixing_coefs.reshape(-1,1),axis=1))
            log_denom = max_vals + np.log(np.sum(np.exp(log_joint-max_vals),axis=1,keepdims=True))
            # for k in range(self.n_components):
            #     log_prior = np.log(self.mixing_coefs[k])
            #     log_pmf_k = log_pmfs[k] # (n_samples,)
            self.gamma = np.exp(log_joint-log_denom)
            # assert False, self.gamma.shape

            # M step
            self.mixing_coefs = np.sum(self.gamma,axis=0) / n_samples
            self.p = self.gamma.T@X / np.sum(self.gamma,axis=0).reshape(-1,1)
    
            n_steps += 1
            if abs(log_likelihood-prev_log_likelihood) < 1e-6:
                break
        print(f'Algorithm stopped after {n_steps} steps.')

    def generate(self,z,size=1):
        return np.random.binomial(n=1,p=self.p[z],size=(size,len(self.p[z])))