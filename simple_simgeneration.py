from pixell import enmap, curvedsky as cs, utils, enplot
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from soapack import interfaces
from scipy.optimize import curve_fit
import pixell.powspec
import sim_generation as simgen
print("Simple sim generation")
fname="/global/cscratch1/sd/jia_qu/maps/2017mask.fits"
mask=enmap.read_map(fname)


map_path='/project/projectdirs/act/data/synced_maps/imaps_2019/'

map0_srfree=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set0_map_srcfree.fits'
ivar0=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set0_ivar.fits'
map0=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set0_map.fits'
map0sz=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set0_map_src_sz.fits'
map1_srfree=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set1_map_srcfree.fits'
ivar1=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set1_ivar.fits'
map1=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set1_map.fits'
map1sz=map_path+'s17_cmb_pa4_f150_nohwp_night_1pass_2way_set1_map_src_sz.fits'

imap0 = enmap.read_map(map0_srfree,)
imap1 = enmap.read_map(map1_srfree,)

ivar0=enmap.read_map(ivar0,)
ivar1=enmap.read_map(ivar1,)



#################################################################################################################################
#create pixel map
shape=imap0.shape
wcs=imap0.wcs
mask = enmap.project(mask,shape,wcs,order=1)
pmap=enmap.pixsizemap(shape,wcs)

map_list=[imap0,imap1]
ivar_list=[ivar0,ivar1]

pol_a=0
pol_b=0
#calculate the flattened power spectrum used to produce simulations
cls=simgen.get_power(map_list,ivar_list, pol_a, pol_b, mask,N=5)
np.savetxt(f'/global/homes/j/jia_qu/sim_gen/flattened{pol_a}{pol_b}.txt',cls)
plot_cl=False
if plot_cl:
	ls = np.arange(cls.size)
	print(cls)
	plt.plot(ls,cls,label='cls')

	plt.axhline(y=1,color='k',ls='--',alpha=1)
	plt.xlabel('$\\ell$')
	plt.ylabel('$C_{\\ell}$')
	plt.yscale('log')
	plt.xscale('log')
	plt.title('Standardized noise power spectrum Nltt')
	plt.savefig('/global/homes/j/jia_qu/sim_gen/cl_noise.png')

sim_list=simgen.generate_sim(ivar_list,cls,10000,0)

a,b=simgen.check_simulation(pol_a,pol_b,map_list,sim_list,ivar_list,mask)
np.savetxt(f'/global/homes/j/jia_qu/sim_gen/simcoadd{pol_a}{pol_b}.txt',a)
np.savetxt(f'/global/homes/j/jia_qu/sim_gen/datacoadd{pol_a}{pol_b}.txt',b)



