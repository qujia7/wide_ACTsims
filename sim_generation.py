import mapsims
from pixell import enmap, curvedsky as cs, utils, enplot
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from soapack import interfaces
from scipy.optimize import curve_fit
import pixell.powspec
from orphics import maps

def eshow(x,fname): 
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, downgrade = 1)
    enplot.write(f"/scratch/r/rbond/jiaqu/plots/{fname}.png",plots)


def project_mask(mask,shape,wcs,fname=None):
	sim_mask = enmap.project(mask,shape,wcs,order=1)
	if fname!=None:
		enmap.write_fits(fname, sim_mask, extra={})

#fitting functions

def rolloff(ell, ell_off=None, alpha=-4, patience=2.):

    """
    Adapted from mapsims
    Get a transfer function T(ell) to roll off red noise at ell <
    ell_off.  ell should be an ndarray.  Above the cut-off,
    T(ell>=ell_off) = 1.  For T(ell<ell_off) will roll off smoothly,
    approaching T(ell) \propto ell^-alpha.  The speed at which the
    transition to the full power-law occurs is set by "patience";
    patience (> 1) is the maximum allowed value of:
                       T(ell) * ell**alpha
                 -----------------------------
                  T(ell_off) * ell_off**alpha
    I.e, supposing you were fighting an ell**alpha spectrum, the
    roll-off in T(ell) will be applied aggressively enough that
    T(ell)*ell**alpha does not rise above "patience" times its value
    at ell_off.
    """
    if ell_off is None or ell_off <= 0:
        return np.ones(ell.shape)
    L2 = ell_off
    L1 = L2 * patience ** (2./alpha)
    x = -np.log(ell / L2) / np.log(L1 / L2)
    beta = alpha * np.log(L1 / L2)
    output = x*0
    output[x<0]  = (-x*x)[x<0]
    output[x<-1] = (1 + 2*x)[x<-1]
    return np.exp(output * beta)   


def rad_fit(x, l0, a):
    return ((l0/x)**-a + 1)

def get_fitting(c_ell):
	#input a power spectrum and get fitted power spectrum
	bounds = ((0, -5), (9000, 1))
	cents=np.arange(len(c_ell))
	#find the floor of the white noise
	w=c_ell[cents > 5000].mean()
	#define the fitting section i.e only l>500
	mask = cents < 500
	ell_fit = cents[~mask]
	c_ell_fit = c_ell[~mask]
	params=np.zeros(3)
	params[:2],_=curve_fit(rad_fit, ell_fit, c_ell_fit/w, p0 = [3000, -4], bounds = bounds)
	params[2]=w	
	fit=rad_fit(cents, params[0], params[1]) * params[2]
	fit[~np.isfinite(fit)]=0
	return rolloff(cents,200,alpha=params[1],patience=1.2)*fit
	#return fitted power spectrum parameters given input powere spectrum

def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def coadd_map(map_list,ivar_list):
	"""return coadded map from splits, input list of maps, where each map only contains one of I,Q or U"""
	map_list=np.array(map_list)
	ivar_list=np.array(ivar_list)
	coadd_map= np.sum(map_list * ivar_list, axis = 0)
	coadd_map/=np.sum(ivar_list, axis = 0)
	coadd_map[~np.isfinite(coadd_map)] = 0

	return enmap.samewcs(coadd_map,map_list[0])
	
def coadd_mapnew(map_list,ivar_list,a):
	"""return coadded map from splits, the map in maplist contains I,Q,U 
	a=0,1,2 selects one of I Q U """
	map_list=np.array(map_list)
	ivar_list=np.array(ivar_list)
	coadd_map= np.sum(map_list[:,a] * ivar_list, axis = 0)
	coadd_map/=np.sum(ivar_list, axis = 0)
	coadd_map[~np.isfinite(coadd_map)] = 0

	return enmap.samewcs(coadd_map,map_list[0])	

def ivar_eff(split,ivar_list):
	"""return effective invers variance map for split i and a list of inverse variance maps.
	Inputs
	splits:integer
	ivar_list:list
	Output
	ndmap with same shape and wcs as individual inverse variance maps.

	"""
	ivar_list=np.array(ivar_list)
	h_c=np.sum(ivar_list,axis=0)
	w=h_c-ivar_list[split]
	weight=1/(1/ivar_list[split]-1/h_c)
	weight[~np.isfinite(weight)] = 0
	weight[weight<0] = 0
	return enmap.samewcs(weight,ivar_list[0])
	


def get_power(map_list,ivar_list, a, b, mask,N=20):
	"""
	Calculate the average coadded flattened power spectrum P_{ab} used to generate simulation for the splits.
	Inputs:
	map_list: list of source free splits
	ivar_list: list of the inverse variance maps splits
	a: 0,1,2 for I,Q,U respectively
	b:0,1,2 for I,Q,U, respectively
	N: window to smooth the power spectrum by in the rolling average.
	mask: apodizing mask

	Output:
	1D power spectrum accounted for w2 from 0 to 10000
	"""
	pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)

	cl_ab=[]
	n = len(map_list)
	#calculate the coadd maps
	if a!=b:
		coadd_a=coadd_mapnew(map_list,ivar_list,a)
		coadd_b=coadd_mapnew(map_list,ivar_list,b)
	else:
		coadd_a=coadd_mapnew(map_list,ivar_list,a)

	for i in range(n):
		print(i)
		if a!=b:
			d_a=map_list[i][a]-coadd_a
			noise_a=d_a*np.sqrt(ivar_eff(i,ivar_list)/pmap)*mask
			alm_a=cs.map2alm(noise_a,lmax=10000)
			d_b=map_list[i][b]-coadd_b
			noise_b=d_b*np.sqrt(ivar_eff(i,ivar_list)/pmap)*mask
			alm_b=cs.map2alm(noise_b,lmax=10000)
			cls = hp.alm2cl(alm_a,alm_b)
			cl_ab.append(cls)
		else:
			d_a=map_list[i][a]-coadd_a
			noise_a=d_a*np.sqrt(ivar_eff(i,ivar_list)/pmap)*mask
			print("generating alms")
			alm_a=cs.map2alm(noise_a,lmax=10000)
			cls = hp.alm2cl(alm_a)
			cl_ab.append(cls)
	cl_ab=np.array(cl_ab)
	sqrt_ivar=np.sqrt(ivar_eff(0,ivar_list)/pmap)
	mask_ivar = sqrt_ivar*0 + 1
	mask_ivar[sqrt_ivar<=0] = 0
	mask=mask*mask_ivar
	mask[mask<=0]=0
	w2=np.sum((mask**2)*pmap) /np.pi / 4.
	power = 1/n/(n-1) * np.sum(cl_ab, axis=0)
	ls=np.arange(len(power))
	power[~np.isfinite(power)] = 0
	power=rolling_average(power, N)
	bins=np.arange(len(power))
	power=maps.interp(bins,power)(ls)
	return power / w2


def generate_sim(ivar_list,cls,lmax,seed):
	"""
	Input: ivar_list: list of inverse variance maps
	cls: flattened 1D power spectrum Pab
	lmax:maximum multipole to generate the simulated maps
	seed: currently a number, need to fix this.
	Returns:
	list of sumulated maps.
	"""
	shape=ivar_list[0].shape
	wcs=ivar_list[0].wcs
	pmap=enmap.pixsizemap(shape,wcs)
	k=len(ivar_list)
	sim_maplist=[]
	for i in range(len(ivar_list)):
		sim_map=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax,spin=0,seed=seed+i)/(np.sqrt(ivar_eff(i,ivar_list)/pmap))
		sim_map[~np.isfinite(sim_map)] = 0
		sim_maplist.append(sim_map)
	return sim_maplist



"""
k=len(map_list)
sim_map0=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax=10000,spin=0,seed=1)/(np.sqrt(ivar_eff(0,ivar_list)/pmap))
sim_map0[~np.isfinite(sim_map0)] = 0
sim_map1=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax=10000,spin=0,seed=2)/(np.sqrt(ivar_eff(1,ivar_list)/pmap))
sim_map1[~np.isfinite(sim_map1)] = 0
sim_map2=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax=10000,spin=0,seed=3)/(np.sqrt(ivar_eff(2,ivar_list)/pmap))
sim_map2[~np.isfinite(sim_map2)] = 0
sim_map3=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax=10000,spin=0,seed=4)/(np.sqrt(ivar_eff(3,ivar_list)/pmap))
sim_map3[~np.isfinite(sim_map3)] = 0
"""

def check_simulation(a,b,map_list,sim_list,ivar_list,mask):
	"""
	Check whether simulated power spectrum P_{ab} is consistent with data. Returns list of (split_sim-coadd,split_data-coadd)
	weighted by the mask*effective_ivar.
	"""
	shape=ivar_list[0].shape
	wcs=ivar_list[0].wcs
	pmap=enmap.pixsizemap(shape,wcs)
	sim_coadd=[]
	data_coadd=[]
	for i in range(len(sim_list)):
		dsim=sim_list[i]-coadd_map(sim_list,ivar_list)
		dsim=dsim*mask*ivar_eff(i,ivar_list)/pmap
		testalm=cs.map2alm(dsim,lmax=10000)
		testalm=testalm.astype(np.complex128) 
		testcl=hp.alm2cl(testalm)
		sim_coadd.append(testcl)
	if a==b:
		for i in range(len(map_list)):
			dataco=map_list[i][a]-coadd_mapnew(map_list,ivar_list,a)
			dataco=dataco*mask*ivar_eff(i,ivar_list)/pmap
			testalm=cs.map2alm(dataco,lmax=10000)
			testalm=testalm.astype(np.complex128) 
			testcl=hp.alm2cl(testalm)
			data_coadd.append(testcl)
	else:
			for i in range(len(map_list)):
				data_a=map_list[i][a]-coadd_mapnew(map_list,ivar_list,a)
				data_a=data_a*mask*ivar_eff(i,ivar_list)/pmap
				data_b=map_list[i][b]-coadd_mapnew(map_list,ivar_list,b)
				data_b=data_b*mask*ivar_eff(i,ivar_list)/pmap
				testalm_a=cs.map2alm(data_a,lmax=10000)
				testalm_a=testalm_a.astype(np.complex128)
				testalm_b=cs.map2alm(data_b,lmax=10000)
				testalm_b=testalm_b.astype(np.complex128)
				testcl=hp.alm2cl(testalm_a,testalm_b)
				data_coadd.append(testcl)
	sim_coadd=np.array(sim_coadd)
	data_coadd=np.array(data_coadd)
	return (sim_coadd,data_coadd)








