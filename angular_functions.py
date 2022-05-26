import numpy as np
from matplotlib import pyplot as plt

def shift_angle(theta, theta_shift):
	theta_shifted = theta + theta_shift
	theta_shifted[(theta_shifted >= 360)] -= 360
	theta_shifted[(theta_shifted < 0)] += 360

	return theta_shifted

def angular_stats(theta):
	""" 
	Calculates angular mean for 
	angles in degrees between 0 & 360.
	Does this by calculating unit vectors
	"""
	x_hat = np.sin( np.radians(theta) )
	y_hat = np.cos( np.radians(theta) )

	x_mean = np.mean(x_hat)
	y_mean = np.mean(y_hat)

	results = {}
	results['mean'] = np.degrees(np.arctan2(x_mean, y_mean))
	if results['mean'] < 0:
		results['mean'] += 360

	theta_shift = (180 - results['mean'])

	results['median'] = np.ma.median(shift_angle(theta, theta_shift)) - theta_shift
	if results['median'] >= 360:
		results['median'] -= 360
	elif results['median'] < 0:
		results['median'] += 360
	results['std'] = np.std(shift_angle(theta, theta_shift))

	return results

def angular_mean(theta):

    stats = angular_stats(theta)

    return stats['mean']

def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, wts=None, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """

    # If lab unit is "compass," with 0 degrees = N, 90 = E, 180 = S, 270 = W
    #  adjust angles to match polar coordinates & change to radians
    if lab_unit=="compass":
        angles = shift_angle((360-angles), 90) * np.pi/180

    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins, weights=wts)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=True, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)

    if lab_unit == "compass":
        label = ['E', 'NE','N','NW','W','SW','S','SE']
        ax.set_xticklabels(label)
    
    return