/**
 * @file methods/decision_tree/mse_gain.hpp
 * @author Archit Saxena
 *
 * The mean absolute deviation gain class, which is a fitness funtion for
 * regression based decision trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more informatio
n.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_MAD_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_MAD_GAIN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

/**
 * The MAD (Mean absolute deviation) gain, is a measure of set purity based on the deviation
 * of dependent values present in the node. This is same thing as negation of
 * deviation of dependent variable from the mean in the node as we will try to
 * maximize this quantity to maximize gain (and thus reduce absolute deviation
 *  of a set).
*/
class MADGain
{
public:
  /**
   * Evaluate the mean absolute deviation gain on the given set of values. Note
   * that gain can be slightly greater than 0 due to floating-point
   * representation issues. Thus if you are checking for perfect fit, be sure
   * to use 'gain >= 0.0'. Not 'gain == 0.0'. The numeric prediction variables
   * should always be of type arma::Row<double> or arma::rowvec.   
   *
   * @param predictors Set of predicted values to evaluate mse gain on.
   * @param weights Weight of labels.
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& predictors,
                         const WeightVecType& weights,
                         const size_t start,
                         const size_t end)
  {

    //Calculate MAD of the un-split node.
    double mad = 0.0;

    if (UseWeights)
    {
      //Sum of all weights and weighted mean.
      double accWeights = 0.0;
      double weightedMean = 0.0;

      // Mean and weight loop: Find weighted mean and sum of weights.
      for (size_t i = start; i <= end; i ++)
      {
        accWeights += weights[i];
        weightedMean += ( predictors[i] * weights[i] ); 
      }

      if (accWeights == 0.0)
        return 0.0;

      weightedMean /= accWeights;

      for (size_t i = start; i <= end; i ++)
      {
        const double f = weights[i] * (std::abs(predictors[i] - weightedMean));
        mad += (f / accWeights); 
      }

    }
    else
    {

      //Mean of data points
      double mean = 0.0;

      //Calculate mean of predictor data points
      for (size_t i = start; i <= end; i ++)
        mean +=  predictors[i];
      mean/=(end-start+1);

      for (size_t i = start; i <= end ; i ++)
      {
        const double f = std::abs(predictors[i] - mean);
        mad += (f / (double) (end - start + 1));
      }

    }
    return -mad;        
  }  

  /**Calcualte over whole vector
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& predictors,
                         const WeightVecType& weights)
  {
    // Corner case: if there are no elements, the impurity is zero.
    if (predictors.n_elem == 0)
      return 0.0;
    return Evaluate<UseWeights>(predictors,weights,0,predictors.n_elem-1);

  }

};
     
} // namespace tree
} // namespace mlpack

#endif
