#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <pybind11/stl.h>

#include <iostream>
#include <numeric>
#include <cmath>

namespace py = pybind11;





auto colored_hist(
  const xt::pytensor<float, 1> & data,
  const xt::pytensor<float, 1> & values,
  const std::array<float, 2> & range,
  const std::size_t n_bins,
  const bool normalize
)
{
    const auto delta = range[1] - range[0];
    const auto bin_width = delta / n_bins;

    auto bin_edges = xt::pytensor<float, 1>::from_shape({n_bins + 1});
    
    auto bin_count = xt::pytensor<float, 1>::from_shape({n_bins});
    auto bin_mean = xt::pytensor<float, 1>::from_shape({n_bins});
    std::fill(bin_count.begin(), bin_count.end(), 0.0);
    std::fill(bin_mean.begin(), bin_mean.end(), 0.0);
    for(auto i=0; i <= n_bins; ++i)
    {
        bin_edges[i] = static_cast<float>(i) * bin_width + range[0];
    }

    auto c = 0 ;
    for(auto i=0; i<data.shape(0); ++i)
    {
        const auto d = data(i);
        if(d >= range[0] && d<= range[1])
        {
            const auto bi = std::size_t(((d - range[0]) / delta) * (n_bins-1) ); 
            ++bin_count(bi);
            bin_mean(bi) += values[i];
            ++c;
        }
    }

    bin_mean /= bin_count;
    if (normalize){
        bin_count /= c;
    }
    return std::make_tuple(bin_count, bin_edges, bin_mean);
}


// Python Module and Docstrings

PYBIND11_MODULE(colored_hist, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        colored hist

        .. currentmodule:: colored_hist

        .. autosummary::
           :toctree: _generate

           colored_hist
    )pbdoc";


    m.def("colored_hist", &colored_hist, py::arg("data"), py::arg("values"), py::arg("range"), py::arg("n_bins") ,py::arg("normalize"));

}
