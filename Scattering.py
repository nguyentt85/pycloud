#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals
from numpy import *

################################################################################
# Description: Model diffraction of electromagnetic waves by spherical         #
#              particles of atmospheric                                        #
# Authors:     Tong Tam Nguyen (nguyen@meteolab.ru)                            #
#              Vladimir V. Chukin (chukin@meteolab.ru)                         #
# Version:     2015-03-12                                                      #
# License:     GNU General Public License 2.0                                  #
################################################################################

# INITIALIZATION WAVELENGTH
class Wavelength(object):
	def __init__(self, WL):
		self._WL = WL
	def getWL(self):
		return self._WL
	def setWL(self, value):
		self._WL = value
	def delWL(self):
		del self._WL
	WL = property(getWL, setWL, delWL, "Property WL")

# INITIALIZATION RADIUS OF PARTICLES
class Radius(object):
	def __init__(self, RP):
		self._RP = RP
	def getRP(self):
		return self._RP
	def setRP(self, value):
		self._RP = value
	def delRP(self):
		del self._RP
	RP = property(getRP, setRP, delRP, "Property RP")

# INITIALIZATION SCATTERING ANGLE
class Theta(object):
	def __init__(self, AGL):
		self._AGL = AGL
	def getAGL(self):
		return self._AGL
	def setAGL(self, value):
		self._AGL = value
	def delAGL(self):
		del self._AGL
	AGL = property(getAGL, setAGL, delAGL, "Property AGL")
	
# INITIALIZATION REFRACTIVE INDEX OF PARTICLES
class Particle(object):
	def __init__(self, PAR):
		self._PAR = PAR
	def getPAR(self):
		return self._PAR
	def setPAR(self, value):
		self._PAR = value
	def delPAR(self):
		del self._PAR
	PAR = property(getPAR, setPAR, delPAR, "Property PAR")

# INITIALIZATION REFRACTIVE INDEX OF ENVIRONMENT
class Environment(object):
	def __init__(self, ENV):
		self._ENV = ENV
	def getENV(self):
		return self._ENV
	def setENV(self, value):
		self._ENV = value
	def delENV(self):
		del self._ENV
	ENV = property(getENV, setENV, delENV, "Property ENV")

# COEFFICIENTS OF MIE THEORY
class Mie(Wavelength, Radius, Theta, Particle, Environment):
	# Complex parameter diffraction
	def getX(self):
		X = 2*pi*self.getRP()*self.getENV()/self.getWL()
		Xc = complex(X,0)
		return Xc
	# Number of coefficients
	def getN(self):
		Ncr = self.getX().real+4*pow(self.getX().real,1.0/3.0)+2.0
		N = int(round(Ncr))+1
		return N
	# Angular function PI
	def getPi(self):
		N = self.getN()
		pp = zeros(N)
		pp[0] = 0.0
		pp[1] = 1.0
		for i in range(2,N,1):
			pp[i] = ((2.0*i-1.0)/(i-1.0))*cos(radians(self.getAGL()))*pp[i-1]-(i/(i-1.0))*pp[i-2]
		return pp
	# Angular function TAU
	def getTau(self):
		N = self.getN()
		tau = zeros(N)
		for i in range(1,N,1):
			tau[i] = i*cos(radians(self.getAGL()))*self.getPi()[i]-(i+1.0)**self.getPi()[i-1]
		return tau
	# Function to canculate coefficients of Mie theory
	def getCoefficients(self):
		N = self.getN()
		chi = zeros(N)
		Dmx = vectorize(complex)(zeros(N+15),zeros(N+15))
		psi = vectorize(complex)(zeros(N),zeros(N))
		xi  = vectorize(complex)(zeros(N),zeros(N))
		N1 = 1.0+0.0j
		x = self.getX()
		# I Riccati-Bessel Function
		psi[0] = complex(sin(x.real),0)
		psi[1] = complex(psi[0].real/x.real-cos(x.real),0)
		for i in range(2,N,1):
			psi[i] = complex(((2.0*(i-1.0)+1.0)/x.real)*psi[i-1].real-psi[i-2].real,0)
		# II Riccati-Bessel Function
		chi[0] = cos(x.real)
		chi[1] = chi[0]/x.real+sin(x.real)
		for i in range(2,N,1):
			chi[i] = ((2.0*(i-1.0)+1.0)/x.real)*chi[i-1]-chi[i-2]
		# III Riccati-Bessel Function
		for i in range(N):
			xi[i] = complex(psi[i].real, -chi[i])
		# Logarithmic derivative of the function
		m = self.getPAR()/self.getENV() # The relative refractive index
		for i in range(N+13,-1,-1):
			n = complex(i+1.0, 0.0)
			Dmx[i] = n/(m*x)-N1/(Dmx[i+1]+n/(m*x))
		# Coefficients
		an = vectorize(complex)(zeros(N),zeros(N))
		bn = vectorize(complex)(zeros(N),zeros(N))
		for i in range(1,N,1):
			n = complex(i, 0.0)
			an[i] = ((Dmx[i]/m+n/x)*psi[i]-psi[i-1])/((Dmx[i]/m+n/x)*xi[i]-xi[i-1])
			bn[i] = ((m*Dmx[i]+n/x)*psi[i]-psi[i-1])/((m*Dmx[i]+n/x)*xi[i]-xi[i-1])
		return an, bn

# CLASS FOR GETTING RESULTS
class Scattering(Mie):
	# Coefficients of Scattering, Absorption, Attenuation and Radar Reflection
	def getAllCoefficients(self):
		x = self.getX().real
		N = self.getN()
		a, b = self.getCoefficients()
		KS = 0
		KE = 0
		KA = 0
		KR = complex(0.0+0.0j)
		for i in range(N):
			KS = KS+(2.0*i+1.0)*(pow(absolute(a[i]),2)+pow(absolute(b[i]),2))
			KE = KE+(2.0*i+1.0)*(a[i].real+b[i].real)
			KR = KR+pow(-1,i)*(2*i+1)*(a[i]-b[i])
		KA = KE-KS
		SS = 2.0*KS/x/x					# Scattering efficiency factor
		AA = 2.0*KA/x/x				    # Absorption efficiency factor
		EE = 2.0*KE/x/x					# Efficiency factor attenuation
		RR = 2.0*pow(absolute(KR),2)/x/x	# Factor in the effectiveness of the radar reflection
		return SS, AA, EE, RR
	# The amplitude of the electric vector		
	def getAmplitudes(self):
		N = self.getN()
		Pi = self.getPi()
		Tau = self.getTau()
		a, b = self.getCoefficients()
		A1 = complex(0.0+0.0j)
		A2 = complex(0.0+0.0j)
		for i in range(N):
			A1 = A1+((2*i+1)/(n*(n+1)))*(a[i]*Pi[i]+b[i]*Tau[i])
			A2 = A2+((2*i+1)/(n*(n+1)))*(b[i]*Pi[i]+a[i]*Tau[i])
		return A1, A2
