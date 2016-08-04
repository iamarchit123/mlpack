/**
 * @file space_split_impl.hpp
 * @author Marcos Pividori
 *
 * Implementation of MidpointSpaceSplit and MeanSpaceSplit, to create a
 * splitting hyperplane considering the midpoint/mean of the values in a certain
 * projection.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_IMPL_HPP

#include "space_split.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType, typename MatType>
template<typename HyperplaneType>
bool MeanSpaceSplit<MetricType, MatType>::SplitSpace(
    const typename HyperplaneType::BoundType& bound,
    const MatType& data,
    const std::vector<size_t>& points,
    HyperplaneType& hyp)
{
  typename HyperplaneType::ProjVectorType projVector;
  double midValue;

  if (!SpaceSplit<MetricType, MatType>::GetProjVector(bound, data, points,
    projVector, midValue))
    return false;

  double splitVal = 0.0;
  for (size_t i = 0; i < points.size(); i++)
    splitVal += projVector.Project(data.col(points[i]));
  splitVal /= points.size();

  hyp = HyperplaneType(projVector, splitVal);

  return true;
}

template<typename MetricType, typename MatType>
template<typename HyperplaneType>
bool MidpointSpaceSplit<MetricType, MatType>::SplitSpace(
    const typename HyperplaneType::BoundType& bound,
    const MatType& data,
    const std::vector<size_t>& points,
    HyperplaneType& hyp)
{
  typename HyperplaneType::ProjVectorType projVector;
  double midValue;

  if (!SpaceSplit<MetricType, MatType>::GetProjVector(bound, data, points,
    projVector, midValue))
    return false;

  hyp = HyperplaneType(projVector, midValue);

  return true;
}

template<typename MetricType, typename MatType>
bool SpaceSplit<MetricType, MatType>::GetProjVector(
    const bound::HRectBound<MetricType>& bound,
    const MatType& data,
    const std::vector<size_t>& /* points */,
    AxisParallelProjVector& projVector,
    double& midValue)
{
  // Get the dimension that has the maximum width.
  size_t splitDim = data.n_rows; // Indicate invalid.
  double maxWidth = -1;

  for (size_t d = 0; d < data.n_rows; d++)
  {
    const double width = bound[d].Width();

    if (width > maxWidth)
    {
      maxWidth = width;
      splitDim = d;
    }
  }

  if (maxWidth <= 0) // All these points are the same.
    return false;

  projVector = AxisParallelProjVector(splitDim);

  midValue = bound[splitDim].Mid();

  return true;
}

template<typename MetricType, typename MatType>
template<typename BoundType>
bool SpaceSplit<MetricType, MatType>::GetProjVector(
    const BoundType& /* bound */,
    const MatType& data,
    const std::vector<size_t>& points,
    ProjVector& projVector,
    double& midValue)
{
  MetricType metric;

  // Efficiently estimate the farthest pair of points in the given set.
  size_t fst = points[rand() % points.size()];
  size_t snd = points[0];
  double max = metric.Evaluate(data.col(fst), data.col(snd));

  for (size_t i = 1; i < points.size(); i++)
  {
    double dist = metric.Evaluate(data.col(fst), data.col(points[i]));
    if (dist > max)
    {
      max = dist;
      snd = points[i];
    }
  }

  std::swap(fst, snd);

  for (size_t i = 0; i < points.size(); i++)
  {
    double dist = metric.Evaluate(data.col(fst), data.col(points[i]));
    if (dist > max)
    {
      max = dist;
      snd = points[i];
    }
  }

  if (max == 0) // All these points are the same.
    return false;

  // Calculate the normalized projection vector.
  projVector = ProjVector(arma::normalise(data.col(snd) - data.col(fst)));

  arma::vec midPoint = (data.col(snd) + data.col(fst)) / 2;

  midValue = projVector.Project(midPoint);

  return true;
}

} // namespace tree
} // namespace mlpack

#endif
