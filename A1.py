#! /usr/bin/python

from datetime import datetime, time, timedelta
from pandas.io.data import DataReader
from pandas.io.data import Options

import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats as ss
import time
import numpy as np
import pdb
import csv

#data = np.loadtxt("JB_2014_JST.csv", delimiter=",")
def questionOne(filename):
	f = open(filename, 'rU')
	csv_f = csv.reader(f)

	price=[]
	size=[]
	open_price=[]
	close_price=[]
	vwap=[]
	for row in csv_f:
		date=datetime.strptime(row[1]+' '+row[2],'%m/%d/%Y %I:%M:%S %p')
		#pdb.set_trace()
		
		#if time(8, 30) <= date.time() <= time(16):
			#if current price is within trading hours

		if not price: #price is empty take care of initial case
			initial_day = date.day
		else:
			if date.day != initial_day:
				# compute open, end, vwap
				# and empty price+size lists, then
				# append price+size for new "day"
				open_price.append(price[0])
				close_price.append(price[-1]) #last element
				prices=np.array(price,dtype=float)
				sizes=np.array(size,dtype=float)
				#pdb.set_trace()
				vwap.append(np.dot(prices,sizes)/sum(sizes))
				del price[:]
				del size[:]
				initial_day = date.day

		price.append(row[3])
		size.append(row[4])

	#pdb.set_trace()
	x = np.arange(len(open_price)) + 1 
	plt.plot(x, np.array(open_price), 'r--', x, np.array(close_price),'bs', x, np.array(vwap),'g^')
	plt.show()

def dayEffect(t1,t2,S,d):
	# Assumes t1,t2,S are strings. d is an int
	bars = DataReader(S, "yahoo", datetime.strptime(t1, '%m/%d/%Y'),datetime.strptime(t2,'%m/%d/%Y'))
	#pdb.set_trace()

	# bars.index[1] Timestamp('2015-01-05 00:00:00')
	#  bars.columns Index([u'Open', u'High', u'Low', u'Close', u'Volume', u'Adj Close'], dtype='object')
	
	returns=[]
	for i in range(len(bars.index)):
		if bars.index[i].day == d:
			returns.append((bars['Close'][i]-bars['Open'][i])/bars['Open'][i])

	print 'The average return for the startegy is ' + str(np.mean(returns)) + ' when trading on every ' + str(d) + ' day' 


#Black and Scholes
def d1(S0, K, r, sigma, T):
	return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
	return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(type,S0, K, r, sigma, T):
	if type=="C":
		return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
	else:
		return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

def impliedVolBisection(S,K,T,r,C,x1,x2):
	tol = 0.000001
	a=x1
	b=x2

	def f(x):
		return BlackScholes('C',S,K,r,x,T) - C

	c = (a+b)/2.0
	while (b-a)/2.0 > tol:
		if f(c) == 0:
			return c
		elif f(a)*f(c) < 0:
			b = c
		else :
			a = c
		c = (a+b)/2.0
		
	return c

def impliedVolNewton(S,K,T,r,C,x0):
	def f(x):
		return BlackScholes('C',S,K,r,x,T) - C

	def df(x,dx):
		return (f(x+dx) - f(x))/dx

	x=x0
	dx=0.000001
	tol=0.000001

	while True:
		x1 = x - f(x)/df(x, dx)
		t = abs(x1 - x)
		if t < tol:
			break
		x = x1

	return x

def getOptionsData(ticker, expiryDate):
	data = Options(ticker, 'yahoo')
	opt_data = data.get_call_data(expiry=expiryDate)
	return opt_data

def questionFive(ticker):
	# compute implied volatility for given ticker for December 2015 calls with 10 strikes
	# assume 1% interest rate
	tic=time.clock()
	data = getOptionsData(ticker, dt.date(2015,12,1))
	bars = DataReader(ticker, 'yahoo', datetime.today() - timedelta(hours=24), datetime.today())
	toc=time.clock()
	print 'Took ' + str(toc-tic) + ' seconds to load data using pandas'
	#pdb.set_trace()

	S = float(bars['Close'])
	T = (12 - float(datetime.today().month))/12
	r = 0.01;

	#pdb.set_trace()

	strikes=[]
	impVols=[]
	num_strikes = data.shape[0]
	tic=time.clock()
	for i in range(num_strikes):
		K = data.index[i][0]
		C = data.iloc[i][0]
		strikes.append(K)
		impVols.append(impliedVolBisection(S,K,T,r,C,0,1))
		#pdb.set_trace()

	toc=time.clock()
	print 'Took ' + str(toc-tic) + ' seconds to complete all IV calculations'
	x=np.array(strikes)
	y=np.array(impVols)
	z = np.polyfit(x, y,2) #fit a second order polynomial to IV
	p = np.poly1d(z)

	#pdb.set_trace()

	xp = np.linspace(strikes[0], strikes[-1], num_strikes)
	plt.plot(x, y, '.', xp, p(xp), '-')
	plt.ylim(0.1,0.6)
	plt.show()

def main():
	#questionOne('JB_2014_JST.csv')
	#dayEffect('01/01/2015','08/31/2015','SPY',3)
	# tic=time.clock()
	# bisecVol = impliedVolBisection(50,50,1,0.02,3.12,0,1)
	# toc=time.clock()
	# print 'Implied Vol using Bisection method is ' + str(bisecVol)
	# print 'Bisection method took ' + str(toc-tic) + ' seconds'

	# tic=time.clock()
	# newtVol = impliedVolNewton(50,50,1,0.02,3.12,0.1)
	# toc=time.clock()
	# print 'Implied Vol using Newton method is ' + str(newtVol)
	# print 'Newton method took ' + str(toc-tic) + ' seconds'

	questionFive('SPY')

main()