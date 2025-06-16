from __future__ import print_function
import copy
import time as pytime
import six
import numpy as np
import iminuit as im
import emcee
import corner
from functools import partial

# assert int(im.__version__[0])<=1 # only support the iMinuit v1

def upperLimit(chi2, fit_par, delta=2.706, tol=0.005, maxtry=30, ignoreErr=False, **mnt_kwds):
    """
    upperLimit calculator, it is optimized to the unreliable minimizer Minuit

    parameters
    ----------
    chi2: chi2 function
    fit_par: str
        the name of the parameter to fit
    delta: float
        the Delta Chi2 to increase, 2.706 for 95% upper limit
    tol: float
        the maximum chi2 difference between the target to accept
    maxtry: int
        the maximum trials to perform
    ignoreErr: bool
        whether to ignore the error in the fittings
    **mnt_kwds: dict
        the parameters passing to the Minuit

    returns
    -------
    medx, dfval, maxx-minx, fval_min
    """
    if not fit_par in im.describe(chi2):
        raise IOError('fit_par %s is not the arg of chi2!' % fit_par)

    assert (delta > 0.) and (tol > 0.) and isinstance(maxtry, int) and (maxtry >= 1)
    nveto = maxtry # the veto algorithm may be problematic

    mnt_kwds_copy = copy.deepcopy(mnt_kwds)
    if 'fix_'+fit_par in mnt_kwds_copy.keys():
        del mnt_kwds_copy['fix_'+fit_par]

    for key, val in mnt_kwds.items():
        if key.startswith('error_') and (~np.isfinite(val)):
            del mnt_kwds_copy[key]

    print('[upperLimit] First fit ...')
    m = doMigradBetter(chi2, **mnt_kwds_copy)
    m.print_fmin()
    x_min, fval_min = m.values[fit_par], m.fval # assume it is the optimized one

    myfit_arg = copy.deepcopy(m.fitarg)
    for key, val in m.fitarg.items():
        if key.startswith('error_') and (~np.isfinite(val)):
            del myfit_arg[key]

    print('-'*50)
    print('[upperLimit] Getting the upperLimit of %s ...' % fit_par)
    if np.isfinite(m.errors[fit_par]):
        dx = m.errors[fit_par]*round(np.sqrt(delta))
    else:
        dx = abs(m.values[fit_par])
    minx, maxx = x_min, x_min + dx
    medx = maxx
    fval = fval_min
    is_veto = True
    for ntry in range(maxtry):
        kwds2 = copy.deepcopy(myfit_arg)
        kwds2['fix_'+fit_par] = True
        kwds2[fit_par] = medx
        mm = doMigradBetter(chi2, **kwds2)

        fval0 = fval
        fval, dfval = mm.fval, mm.fval-(fval_min+delta)
        print('%02i  %g  %g  %g  %g' % (ntry, medx, dfval, minx, maxx))
        if (fval < fval_min-tol):
            if ignoreErr:
                print('WARNING: fval_min-fval=%.5g>tol(%.5g), better_value=%s'% (fval_min-fval, tol, medx))
            else:
                raise ValueError('[upperLimit] fval_min-fval=%.5g>tol(%.5g), better_value=%s'% (fval_min-fval, tol, medx))

        if dfval == 0.:
            maxx, minx = medx, medx
            break
        elif abs(dfval) <= tol:
            break
        elif (maxx-minx)<(maxx+minx)/2.*1e-6:
            print('WARNING: the error can not be decreased any more')
            break
        elif is_veto and dfval < 0.:
            if fval-fval0 < 1.e-2:
                dx *= 2.
            else:
                dx = -dfval*(maxx-minx)/(fval-fval0)
            minx, maxx = maxx, maxx + dx
            medx = maxx
        elif dfval > 0.:
            is_veto = False
            minx, maxx = minx, medx
            if (minx != 0.) and (maxx/minx > 1e2):
                medx = np.sqrt(minx*maxx)
            else:
                medx = (maxx+minx)/2.
        elif dfval < 0.:
            minx, maxx = medx, maxx
            if (minx != 0.) and (maxx/minx > 1e2):
                medx = np.sqrt(minx*maxx)
            else:
                medx = (maxx+minx)/2.

    return medx, dfval, maxx-minx, fval_min

## MCMC related
def lnprob_default(x, chi2, lnprior=None, bounds=None):
    # deal with the priors
    if callable(lnprior):
        lp = lnprior(*x)
        if not np.isfinite(lp):
            return -np.inf
    else:
        lp = 0.
    if bounds is not None:
        for xx, bd in zip(x, bounds):
            if bd is None:
                continue
            else:
                if xx<bd[0] or xx>bd[1]:
                    return -np.inf
    lk = -chi2(*x)/2.
    return lk+lp

class MyMCMC(object):
    """
    The wrapper of emcee.EnsembleSampler. You can use it almost the same way as iminuit.Minuit.
    """

    def __init__(self, chi2, lnprior=None, nwalker=None, pool=None, **kwds):
        """
        lnprior is flat in default. If lnprior is not None, the limit and fix
        of the parameters are useless!
        """
        self.parameters = im.describe(chi2)
        self.ndim = len(self.parameters)
        self.nwalker = 2*self.ndim
        if isinstance(nwalker, int):
            self.nwalker = max(nwalker, self.nwalker)

        has_bounds = False
        self.init_guess, self.bound_range = [], []
        for pname in self.parameters:
            if kwds.get('fix_%s'%pname, False):
                if pname in kwds:
                    pval = kwds[pname]
                else:
                    raise IOError('Please input the initial value of "%s"!'%pname)
                self.bound_range.append((pval, pval))
                self.init_guess.append(pval)
                has_bounds = True
            else:
                bd = kwds.get('limit_%s'%pname, None)
                if bd is not None:
                    has_bounds = True
                self.bound_range.append(bd)

                if bd is None:
                    pval = kwds.get(pname, 0.)
                else:
                    bd_isfinite = np.isfinite(bd)
                    assert bd_isfinite.size == 2
                    if not bd_isfinite.any():
                        pval = kwds.get(pname, 0.)
                    elif bd_isfinite.all():
                        pval = kwds.get(pname, np.mean(bd))
                    elif bd_isfinite[0]:
                        pval = kwds.get(pname, bd[0])
                    else:
                        pval = kwds.get(pname, bd[1])
                self.init_guess.append(pval)

        if has_bounds:
            self.lnprob = partial(lnprob_default, chi2=chi2, lnprior=lnprior, bounds=self.bound_range)
        else:
            self.lnprob = partial(lnprob_default, chi2=chi2, lnprior=lnprior)
        self.sampler = emcee.EnsembleSampler(self.nwalker, self.ndim, self.lnprob, pool=pool)

        self._mcinit = {}
        pos0 = []
        for val, bd in zip(self.init_guess, self.bound_range):
            if bd is None:
                vrand = val + 1e-4*max(val, 1.)*np.random.randn(self.nwalker)
            else:
                vmin, vmax = bd
                if np.abs(val) < 1.e-2:
                    vrand = val + 1e-6*(vmax-vmin)*np.random.randn(self.nwalker)
                else:
                    vrand = (1.+1e-3*np.random.randn(self.nwalker)) * val
                vrand = np.clip(vrand, a_min=vmin, a_max=vmax)
            pos0.append(vrand)
        pos0 = np.transpose(pos0)
        self._mcinit['pos0'] = pos0

        self._burnin = None
        self.samples = None
        self.nstep = None

        self._fixed_flags = None

    def check_converge(self):
        if self.sampler.flatchain.size == 0:
            raise RuntimeError('Please do sampling first!')

        newchain = self.sampler.chain[:, :, ~self._fixed_flags] # (nchain, nsample, nparam)
        newparam = []
        for idx, flag in enumerate(self._fixed_flags):
            if not flag:
                newparam.append(self.parameters[idx])

        conv_dict = {}

        # Autocorrelation (adapted from emcee.autocorr)
        c = 5 # step size for the window search

        n_w, n_t, n_d = newchain.shape
        windows, tau_est = [], []
        n_t_next2 = 2**np.ceil(np.log2(n_t)).astype(int)
        for i_d in range(n_d):
            f = np.zeros(n_t)
            for i_w in range(n_w):
                arr = newchain[i_w, :, i_d]
                farr = np.fft.fft(arr-np.mean(arr), n=2*n_t_next2)
                acfarr = np.fft.ifft(farr*np.conjugate(farr))[:n_t].real
                f += acfarr/acfarr[0]
            f /= n_w
            taus = 2.*np.cumsum(f)-1.

            mm = np.arange(len(taus))<c*taus
            if mm.any():
                win = np.argmin(mm)
            else:
                win = len(taus)-1
            windows.append(win)
            tau_est.append(taus[win])

        actdic, infos = {}, []
        for pname, myact in zip(newparam, tau_est):
            actdic[pname] = myact
            infos.append('(%s)%.2f[%.1f]'%(pname, myact, n_t/myact))
        conv_dict['Autocorrelation'] = actdic
        print('Autocorrelation Time:', ' '.join(infos))

        # Gelman-Rubin diagnostic
        nsample = newchain.shape[1]
        thetaj_mean = newchain.mean(axis=1) # (nchain, nparam)
        sj_sq = newchain.std(axis=1, ddof=1) # (nchain, nparam)
        W = sj_sq.mean(axis=0)
        B = nsample*thetaj_mean.std(axis=0, ddof=1)
        var = (1.-1./nsample)*W+B/nsample
        R = np.sqrt(var/W)

        grdict, infos = {}, []
        for pname, rr in zip(newparam, R):
            grdict[pname] = rr
            infos.append('(%s)%.2f'%(pname, rr))
        conv_dict['R'] = grdict
        print('Gelman-Rubin Diagnostic:', ' '.join(infos))

        return conv_dict

    def run(self, nstep=1000, burnin=0.3):
        """
        nstep per walker.
        You can set burnin=0 and run it again after the first run to make more samples.
        """
        assert isinstance(burnin, float) and (0. <= burnin < 1.)

        # burn in
        nburnin = int(burnin*nstep)
        if 0. < burnin < 1.:
            if int(emcee.__version__[0]) < 3:
                pos, probm, state = self.sampler.run_mcmc(N=nburnin, **self._mcinit)
            else:
                init_state = emcee.State(coords=self._mcinit['pos0'], random_state=self._mcinit.get('rstate0', None))
                pos, probm, state = self.sampler.run_mcmc(nsteps=nburnin, initial_state=init_state,
                        skip_initial_state_check=True)
            self._mcinit['pos0'] = pos
            self._mcinit['rstate0'] = state

            self.sampler.reset()
        else:
            print('[WARN] DO NOT BURN-IN THE CHAINS!!')

        # run mcmc
        if int(emcee.__version__[0]) < 3:
            pos, probm, state = self.sampler.run_mcmc(N=nstep, **self._mcinit)
        else:
            init_state = emcee.State(coords=self._mcinit['pos0'], random_state=self._mcinit.get('rstate0', None))
            pos, probm, state = self.sampler.run_mcmc(nsteps=nstep, initial_state=init_state,
                    skip_initial_state_check=True)
        self._mcinit['pos0'] = pos
        self._mcinit['rstate0'] = state

        self.samples = self.sampler.flatchain.copy()
        print('Mean acceptance fraction: %.3f' % np.mean(self.sampler.acceptance_fraction))

        mins, maxs = self.samples.min(axis=0), self.samples.max(axis=0)
        self._fixed_flags = (mins == maxs)
        if self._fixed_flags.all():
            raise RuntimeError('All parameters are fixed?!')

        self.nstep = nstep
        self._burnin = burnin

    def reset(self):
        """
        Reset the mcmc object.
        """
        self.sampler.reset()

    @property
    def args(self):
        """
        The best fit parameters.
        """
        best_index = np.argmax(self.sampler.flatlnprobability)
        best_param = self.sampler.flatchain[best_index, :]
        return best_param

    @property
    def fval(self):
        """
        The log Posterior for the best-fit parameters.
        """
        return np.max(self.sampler.flatlnprobability)

    @property
    def args_post(self):
        """
        The mean value.
        """
        return self.samples.mean(axis=0)

    def _make_dict(self, pvals):
        outdic = {}
        for pname, pval in zip(self.parameters, pvals):
            outdic[pname] = pval
        return outdic

    @property
    def values(self):
        """
        The best fit parameters saved in the dict.
        """
        return self._make_dict(self.args)

    @property
    def errors(self):
        perrs = np.sqrt(((self.samples-self.args.reshape(1, -1))**2).sum(axis=0)\
              / (self.samples.shape[0]-1.))
        return self._make_dict(perrs)

    @property
    def values_post(self):
        """
        The mean value of parameters in the burn-in samples.
        """
        return self._make_dict(self.args_post)

    @property
    def errors_post(self):
        """
        The RMS of parameters in the burn-in samples.
        """
        return self._make_dict(self.samples.std(axis=0, ddof=1))

    @property
    def burnin(self):
        """
        The fraction of samples in the begining of each chain are removed.
        """
        return self._burnin

    def _np_matrix(self, best_pvals, correlation=False):
        sample_diff = (self.samples - best_pvals.reshape(1, -1))[:, ~self._fixed_flags]
        Sxy = np.matmul(sample_diff.T, sample_diff)/(self.samples.shape[0]-1.)
        if correlation:
            Sxx = Sxy.diagonal()
            return Sxy/np.sqrt(np.matmul(Sxx.reshape(-1, 1), Sxx.reshape(1, -1)))
        else:
            return Sxy

    def np_matrix(self, correlation=False):
        """
        No fixed parameters will apear in the matrix!
        """
        return self._np_matrix(self.args, correlation=correlation)

    def np_matrix_post(self, correlation=False):
        """
        Error or correlation matrix in numpy array format.
        No fixed parameters will apear in the matrix!
        """
        return self._np_matrix(self.args_post, correlation=correlation)

    def _make_covariance(self, mycovar):
        myidx_i = 0
        mydict = {}
        for iflag, ipname in zip(self._fixed_flags, self.parameters):
            if iflag:
                continue

            myidx_j = 0
            for jflag, jpname in zip(self._fixed_flags, self.parameters):
                if jflag:
                    continue
                mydict[(ipname, jpname)] = mycovar[myidx_i, myidx_j]
                myidx_j += 1
            myidx_i += 1
        return mydict

    @property
    def covariance(self):
        return self._make_covariance(self.np_matrix(correlation=False))

    @property
    def covariance_post(self):
        return self._make_covariance(self.np_matrix_post(correlation=False))

    def get_percentile(self, cdf=0.5):
        """
        Get the percentile of each parameters using these samples.
        """
        vals = []
        for idim, flag in enumerate(self._fixed_flags):
            if flag:
                _val = self.samples[0, idim]
            else:
                _val = np.percentile(self.samples[:, idim], 100*cdf)
            vals.append(_val)
        return np.atleast_1d(vals)

    def save_samples(self, outfile):
        """
        Save the lnprobability and the corresponding parameters.
        """
        if outfile.endswith('.npz'):
            np.savez_compressed(outfile, parameters=self.parameters,
                    lnprob=self.sampler.flatlnprobability, sampl=self.samples)
        else:
            np.savetxt(outfile, np.c_[self.sampler.flatlnprobability, self.samples],
                   delimiter='  ', header='lnprob #@#' + '  '.join(self.parameters))

    def save_cornerplot(self, cornerfile=None, label_pnames=None, **kwds):
        """
        Get the corner plot of parameters.
        !! No fixed parameters will apear in the corner plot!
        LaTeX pnames can be input through label_pnames, otherwise the parameters of the
        loglikelihood are used. Please note that label_pnames should be the same order as
        self.parameters.
        You can pass more parameters to corner.corner.
        """
        if label_pnames is None:
            mypnames = self.parameters
        else:
            assert len(label_pnames) == len(self.parameters)
            mypnames = label_pnames

        return make_cornerplot(self.samples, mypnames, pvalsplot=self.args,
            outfile=cornerfile, **kwds)

def read_samples(infiles):
    """
    read the saved sample data from the input file

    NOTE:
    - no attempt is done to ensure the parameters or their orders are the same
      between different files

    parameters
    ----------
    infiles: str or list
        a list of the names of input sample files only support txt and npz now

    returns
    -------
    outdata: dict
        the compiled data of the data just read from the input files
        - samples: doubles, shape: (nsampl, nparams)
            the array comprise of the values of each sample
        - lnprobs: doubles, shape: (nsampl,)
            the lnprobability of each sample
        - pnames: strs, shape: (nsampl,)
            the name of each parameter
    """
    if isinstance(infiles, str):
        infiles = [infiles]

    sampls, lnprobs = [], []
    parameters = None
    for infile in infiles:
        if infile.endswith('.npz'):
            indat = np.load(infile)
            sampl = indat['sampl']
            lnprob = indat['lnprob']
            if parameters is None:
                parameters = indat['parameters'].tolist()
        else:
            indat = np.loadtxt(infile)
            lnprob = indat[:, 0]
            sampl = indat[:, 1:]
            if parameters is None:
                with open(infile, 'r') as fh:
                    hdr = fh.readline().strip()
                    parameters = hdr.split('#@#')[-1].split()
        sampls.append(sampl)
        lnprobs.append(lnprob)
    return dict(samples=np.vstack(sampls), lnprobs=np.vstack(lnprobs), pnames=parameters)

def make_cornerplot(samples, pnames, pvalsplot=None, outfile=None, dropidx=None, **kwds):
    """
    make the corner plot with the input MC samples

    parameters
    ----------
    samples: doubles, shape: (nsampl, nparams)
        the array comprise of the values of each sample
    pnames: strs, shape: (nsampl,)
        the name of each parameter
    pvalsplot: doubles or None, shape: (nparams,)
        the values added to the figure
    outfile: str or None
        - str: the name of the output file
        - None: nothing will be saved
    dropidx: int, list or None
        the index of the parameter (in the pnames) to drop
        - None: do not drop any column
        - int: drop the i-th column
        - list: drop all the columns in the list
    **kwds:
        more parameters that passes to the corner.corner

    returns
    -------
    fig: pyplot.Figure
        the matplotlib figure object
    """
    nparams = samples.shape[1]
    assert nparams==len(pnames) and (pvalsplot is None or len(pvalsplot)==nparams)

    pmins = samples.min(axis=0)
    pmaxs = samples.max(axis=0)
    is_fixed = (pmins==pmaxs)
    if isinstance(dropidx, (int, list)):
        is_fixed[dropidx] = True

    idx = 0
    fpar_idx, newpars = [], []
    if pvalsplot is None:
        mybestfits = None
    else:
        mybestfits = []
    for myflag, pname in zip(is_fixed, pnames):
        if not myflag:
            fpar_idx.append(idx)
            newpars.append(pname)
            if mybestfits is not None:
                mybestfits.append(pvalsplot[idx])
        idx += 1

    fig = corner.corner(
        samples.take(fpar_idx, axis=1), labels=newpars, quantiles=[0.16, 0.5, 0.84],
        truths=mybestfits, truth_color='red', use_math_text=True,
        show_titles=True, title_kwargs={"fontsize": 12}, **kwds
    )
    if outfile is not None:
        fig.savefig(outfile)
    return fig
