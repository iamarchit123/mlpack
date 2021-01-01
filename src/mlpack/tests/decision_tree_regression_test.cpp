/**
 * @file tests/decision_tree_regression_test.cpp
 * @author Archit Saxena 
 *
 * Tests for the DecisionTree regresion class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/best_binary_numeric_split.hpp>
#include <mlpack/methods/decision_tree/all_categorical_split.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/all_dimension_select.hpp>
#include <mlpack/methods/decision_tree_regression/mse_gain.hpp>
#include <mlpack/methods/decision_tree_regression/mad_gain.hpp>
#include <mlpack/methods/decision_tree_regression/mad_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::distribution;

/**
 * Make sure the MSE gain is zero when the prediction values are same.
 */
TEST_CASE("MSEGainPerfectTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec predictors;
  predictors.ones(10);
  REQUIRE(MSEGain::Evaluate<false>(predictors, weights) ==
          Approx(0.0).margin(1e-5));
}


/**
 * The MSE gain of an empty vector is 0.
 */
TEST_CASE("MSEGainEmptyTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec predictors;
  REQUIRE(MSEGain::Evaluate<false>(predictors, weights) ==
          Approx(0.0).margin(1e-5));

  REQUIRE(MSEGain::Evaluate<true>(predictors, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * The MSE gain is -( b - a) ^ 2 / 4 for k points evenly split between a and
 *  b.
 */
TEST_CASE("MSEGainEvenSplit", "[DecisionTreeRegressionTest]")
{
  // Try with many different number of points.
  for (size_t c = 2; c < 30; c+=2)
  {
    const size_t numpoints = 100 * c;
    arma::rowvec predictors(numpoints);
    arma::rowvec weights(numpoints);
    for (size_t i = 0; i < numpoints; i+=2)
    {
      predictors[i] = (double)c;
      weights[i] = 1;
      predictors[i+1]= (double)c+2.0;
      weights[i+1]= 1;
    }

    // Calculate MSE gain and make sure it is correct.
    REQUIRE(MSEGain::Evaluate<false>(predictors, weights) ==
        Approx(-1.0 ).epsilon(1e-7));
    REQUIRE(MSEGain::Evaluate<true>(predictors, weights) ==
        Approx(-1.0 ).epsilon(1e-7));
  }
}


/**
 * To make sure the MSE gain can been cacluate proporately with weight.
 */
TEST_CASE("MSEGainWithWeight", "[DecisionTreeRegressionTest]")
{
  arma::rowvec predictors(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 5; ++i)
  {
    predictors[i] = 0.0;
    weights[i] = 0.3;
  }
  for (size_t i = 5; i < 10; ++i)
  {
    predictors[i] = 1.0;
    weights[i] = 0.7;
  }

  REQUIRE(MSEGain::Evaluate<true>(predictors, weights) ==
      Approx(-0.21).epsilon(1e-7));
}

/**
 * Make sure the MAD gain is zero when the prediction values are same.
 */
TEST_CASE("MADGainPerfectTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec predictors;
  predictors.zeros(0);

  REQUIRE(MADGain::Evaluate<false>(predictors, weights) ==
          Approx(0.0).margin(1e-5));
}


/**
 * The MAD gain of an empty vector is 0.
 */
TEST_CASE("MADGainEmptyTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec predictors;
  REQUIRE(MADGain::Evaluate<false>(predictors, weights) ==
          Approx(0.0).margin(1e-5));

  REQUIRE(MADGain::Evaluate<true>(predictors, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * The MAD gain is -( b - a) / 2 for k points evenly split between a and
 *  b.
 */
TEST_CASE("MADGainEvenSplit", "[DecisionTreeRegressionTest]")
{
  // Try with many different number of points.
  for (size_t c = 2; c < 30; c+=2)
  {
    const size_t numpoints = 100 * c;
    arma::rowvec predictors(numpoints);
    arma::rowvec weights(numpoints);
    for (size_t i = 0; i < numpoints; i+=2)
    {
      predictors[i] = (double)c;
      weights[i] = 1;
      predictors[i+1]= (double)c+2.0;
      weights[i+1]= 1;
    }

    // Calculate MAD gain and make sure it is correct.
    REQUIRE(MADGain::Evaluate<false>(predictors, weights) ==
        Approx(-1.0 ).epsilon(1e-7));
    REQUIRE(MADGain::Evaluate<true>(predictors, weights) ==
        Approx(-1.0 ).epsilon(1e-7));
  }
}


/**
 * To make sure the MAD gain can been cacluate proporately with weight.
 */
TEST_CASE("MADGainWithWeight", "[DecisionTreeRegressionTest]")
{
  arma::rowvec predictors(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 5; ++i)
  {
    predictors[i] = 0.0;
    weights[i] = 0.3;
  }
  for (size_t i = 5; i < 10; ++i)
  {
    predictors[i] = 1.0;
    weights[i] = 0.7;
  }

  REQUIRE(MADGain::Evaluate<true>(predictors, weights) ==
      Approx(-0.42).epsilon(1e-7));
}

/**
 * Check that the BestBinaryNumericSplit will split on an obviously splittable
 * dimension.
 */
TEST_CASE("BestBinaryNumericSplitSimpleSplitTest", "[DecisionTreeRegressionTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::rowvec predictors("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(predictors.n_elem);
  weights.ones();

  arma::vec classProbabilities;
  BestBinaryNumericSplit<MADGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = MADGain::Evaluate<false>(predictors, weights);
  const double gain = BestBinaryNumericSplit<MADGain>::SplitIfBetter<false>(
      bestGain, values, predictors, weights, 3, 1e-7, classProbabilities,
      aux);
  const double weightedGain =
      BestBinaryNumericSplit<MADGain>::SplitIfBetter<true>(bestGain, values,
      predictors, weights, 3, 1e-7, classProbabilities, aux);


  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Make sure weight works and is not different than the unweighted one.
  REQUIRE(gain == weightedGain);

  // The split is perfect, so we should be able to accomplish a gain of 0.
  REQUIRE(gain == Approx(0.0).margin(1e-7));

  // The class probabilities, for this split, hold the splitting point, which
  // should be between 4 and 5.
  REQUIRE(classProbabilities.n_elem == 1);
  REQUIRE(classProbabilities[0] > 0.4);
  REQUIRE(classProbabilities[0] < 0.5);
}

/**
 * Check that the BestBinaryNumericSplit won't split if not enough points are
 * given.
 */
TEST_CASE("BestBinaryNumericSplitMinSamplesTest", "[DecisionTreeRegressionTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::rowvec predictors("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(predictors.n_elem);

  arma::vec classProbabilities;
  BestBinaryNumericSplit<MSEGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(predictors, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, predictors, weights, 8, 1e-7, classProbabilities,
      aux);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      BestBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, values,
      predictors,  weights, 8, 1e-7, classProbabilities, aux);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * Check that the BestBinaryNumericSplit doesn't split a dimension that gives no
 * gain.
 */
TEST_CASE("BestBinaryNumericSplitNoGainTest", "[DecisionTreeTest]")
{
  arma::vec values(100);
  arma::rowvec predictors(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    values[i] = i;
    predictors[i] = 0;
    values[i + 1] = i;
    predictors[i + 1] = 1;
  }

  arma::vec classProbabilities;
  BestBinaryNumericSplit<MSEGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(predictors, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, predictors, weights, 10, 1e-7, classProbabilities,
      aux);

  // Make sure there was no split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * Check that the AllCategoricalSplit will split when the split is obviously
 * better.
 */
TEST_CASE("AllCategoricalSplitSimpleSplitTest", "[DecisionTreeRegressionTest]")
{
  arma::vec values("0 0 0 1 1 1 2 2 2 3 3 3");
  arma::rowvec predictors("10 10 10 20 20 20 10 10 10 20 20 20");
  arma::rowvec weights(predictors.n_elem);
  weights.ones();

  arma::vec classProbabilities;
  AllCategoricalSplit<MSEGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(predictors, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, 4, predictors, weights, 3, 1e-7, classProbabilities,
      aux);
  const double weightedGain =
      AllCategoricalSplit<MSEGain>::SplitIfBetter<true>(bestGain, values, 4,
      predictors, weights, 3, 1e-7, classProbabilities, aux);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Since the split is perfect, make sure the new gain is 0.
  REQUIRE(gain == Approx(0.0).margin(1e-7));

  REQUIRE(gain == weightedGain);

  // Make sure the class probabilities now hold the number of children.
  REQUIRE(classProbabilities.n_elem == 1);
  REQUIRE((size_t) classProbabilities[0] == 4);
}

/**
 * Make sure that AllCategoricalSplit respects the minimum number of samples
 * required to split.
 */
TEST_CASE("AllCategoricalSplitMinSamplesTest", "[DecisionTreeTest]")
{
  arma::vec values("0 0 0 1 1 1 2 2 2 3 3 3");
  arma::rowvec predictors("10 10 10 20 20 20 30 30 30 40 40 40");
  arma::rowvec weights(predictors.n_elem);
  weights.ones();

  arma::vec classProbabilities;
  AllCategoricalSplit<MADGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = MADGain::Evaluate<false>(predictors, weights);
  const double gain = AllCategoricalSplit<MADGain>::SplitIfBetter<false>(
      bestGain, values, 4, predictors, weights, 4, 1e-7, classProbabilities,
      aux);

  // Make sure it's not split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * Check that no split is made when it doesn't get us anything.
 */
TEST_CASE("AllCategoricalSplitNoGainTest", "[DecisionTreeTest]")
{
  arma::vec values(300);
  arma::rowvec predictors(300);
  arma::rowvec weights = arma::ones<arma::rowvec>(300);

  for (size_t i = 0; i < 300; i += 3)
  {
    values[i] = int(i / 3) % 10;
    predictors[i]= 10;
    values[i + 1] = int(i / 3) % 10;
    predictors[i + 1] = 20;
    values[i + 2] = int(i / 3) % 10;
    predictors[i + 2] = 30;
  }

  arma::vec classProbabilities;
  AllCategoricalSplit<MSEGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(predictors, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, 10, predictors, weights, 10, 1e-7,
      classProbabilities, aux);
  const double weightedGain =
      AllCategoricalSplit<MSEGain>::SplitIfBetter<true>(bestGain, values, 10,
      predictors, weights, 10, 1e-7, classProbabilities, aux);

  // Make sure that there was no split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
  REQUIRE(classProbabilities.n_elem == 0);
}
