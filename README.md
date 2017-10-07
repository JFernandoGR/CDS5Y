# CDS5Y
CDS Implied Ratings

In this repository, you will find the codes used in the calculation of Implied Ratings from 5Y Credit Default Swaps, 
using classification trees and Kou & Varotto (2008) methodology. It is subject to change as new improvements are registered.
All functions used are available in Matlab 2015a version.

Main Code: Bond Ratings.m

Inputs: 

(I) Reclassification.xlsx: Categories established by Marmi, Nasigh and Regoli (2014) for reducing the number of classes.
(II) Sovereign Bond Ratings.xlsx: There are several sheets in this file:

(a) Bond Rating: A database of historical changes of CRA ratings.
(b) Regions: A list of countries with their respective associated region to estimate transition probabilities.
(c) CDS-5Y: A historical database of CDS 5Y until May-2017, used in the calculation of implied ratings.
(d) Classification Tests: A historical database of CDS 5Y series of countries with incomplete information.

Functions:

(I) CompleteValues.m: Complete missing data with known values.
(II) KV Penalty Function.m: Function used to extract CDS limits in the Kou-Varotto (2008) methodology.
(III) countmember.m: C = COUNTMEMBER(A,B) counts the number of times the elements of array A are present in array B, so that C(k) equals the number of occurences of A(k) in B. A may contain non-unique elements. C will have the same size as A. 
A and B should be of the same type, and can be cell array of strings. 

If you have specific questions, please send me a message: jairo.gudino@correounivalle.edu.co.
