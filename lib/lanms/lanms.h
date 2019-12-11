#pragma once

#include "clipper/clipper.hpp"
#include <iostream>

#include <vector>
#include <iterator>
#include <algorithm>
#include <iomanip>


// locality-aware NMS
namespace lanms {

	namespace cl = ClipperLib;
    // 多边形结构体
	struct Polygon {
		cl::Path poly;
		double score;
		void print_polygon(){
			for(auto iter = poly.begin(); iter != poly.end(); iter++)
			{
				std::cout << std::setprecision(11) << iter->X << " " << iter->Y << " ";
			}
			std::cout << score << std::endl;
		}
	};

    //
	float paths_area(const ClipperLib::Paths &ps) {
		float area = 0;
		for (auto &&p: ps)
			area += cl::Area(p);
		return area;
	}

    // 多边形iou
	float poly_iou(const Polygon &a, const Polygon &b) {
		cl::Clipper clpr;
		clpr.AddPath(a.poly, cl::ptSubject, true);
		clpr.AddPath(b.poly, cl::ptClip, true);

		cl::Paths inter, uni;
		clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
		clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);

		auto inter_area = paths_area(inter),
			 uni_area = paths_area(uni);
		return std::abs(inter_area) / std::max(std::abs(uni_area), 1.0f);
	}

    // 判断是否进行合并
	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold) {
		return poly_iou(a, b) > iou_threshold;
	}

	/**
	 * Incrementally merge polygons
	 */
	class PolyMerger {
		public:
			PolyMerger(): score(0), nr_polys(0) {
				memset(data, 0, sizeof(data));
			}

			/**
			 * Add a new polygon to be merged.
			 */
			void add(const Polygon &p_given) {
				Polygon p;
				if (nr_polys > 0) {
					// vertices of two polygons to merge may not in the same order;
					// we match their vertices by choosing the ordering that
					// minimizes the total squared distance.
					// see function normalize_poly for details.
					p = normalize_poly(get(), p_given);
				} else {
					p = p_given;
				}
				assert(p.poly.size() == 4);
				auto &poly = p.poly;
				auto s = p.score;
				data[0] += poly[0].X * s;
				data[1] += poly[0].Y * s;

				data[2] += poly[1].X * s;
				data[3] += poly[1].Y * s;

				data[4] += poly[2].X * s;
				data[5] += poly[2].Y * s;

				data[6] += poly[3].X * s;
				data[7] += poly[3].Y * s;

				score += p.score;

				nr_polys += 1;
			}

			inline std::int64_t sqr(std::int64_t x) { return x * x; }

			Polygon normalize_poly(
					const Polygon &ref,
					const Polygon &p) {

				std::int64_t min_d = std::numeric_limits<std::int64_t>::max();
				size_t best_start = 0, best_order = 0;

				for (size_t start = 0; start < 4; start ++) {
					size_t j = start;
					std::int64_t d = (
							sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 0) % 4].X)
							+ sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 0) % 4].Y)
							+ sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 1) % 4].X)
							+ sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 1) % 4].Y)
							+ sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 2) % 4].X)
							+ sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 2) % 4].Y)
							+ sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 3) % 4].X)
							+ sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 3) % 4].Y)
							);
					if (d < min_d) {
						min_d = d;
						best_start = start;
						best_order = 0;
					}

					d = (
							sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 3) % 4].X)
							+ sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 3) % 4].Y)
							+ sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 2) % 4].X)
							+ sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 2) % 4].Y)
							+ sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 1) % 4].X)
							+ sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 1) % 4].Y)
							+ sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 0) % 4].X)
							+ sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 0) % 4].Y)
						);
					if (d < min_d) {
						min_d = d;
						best_start = start;
						best_order = 1;
					}
				}

				Polygon r;
				r.poly.resize(4);
				auto j = best_start;
				if (best_order == 0) {
					for (size_t i = 0; i < 4; i ++)
						r.poly[i] = p.poly[(j + i) % 4];
				} else {
					for (size_t i = 0; i < 4; i ++)
						r.poly[i] = p.poly[(j + 4 - i - 1) % 4];
				}
				r.score = p.score;
				return r;
			}

			Polygon get() const {
				Polygon p;

				auto &poly = p.poly;
				poly.resize(4);
				auto score_inv = 1.0d / std::max(1e-8d, score);
				poly[0].X = data[0] * score_inv;
				poly[0].Y = data[1] * score_inv;
				poly[1].X = data[2] * score_inv;
				poly[1].Y = data[3] * score_inv;
				poly[2].X = data[4] * score_inv;
				poly[2].Y = data[5] * score_inv;
				poly[3].X = data[6] * score_inv;
				poly[3].Y = data[7] * score_inv;

				assert(score > 0);
				p.score = score;

				return p;
			}

		private:
			double data[8];
			double score;
			std::int32_t nr_polys;
	};


	float left_merge(float a_x, float b_x, float a_score, float b_score){
	    float max_score = std::max(a_score, b_score);
	    float min_score = std::min(a_score, b_score);
	    float left_x = std::min(a_x, b_x);
	    float right_x = std::max(a_x, b_x);

	    return left_x * max_score + right_x * min_score;
	}

	float right_merge(float a_x, float b_x, float a_score, float b_score){
	    float max_score = std::max(a_score, b_score);
	    float min_score = std::min(a_score, b_score);
	    float left_x = std::min(a_x, b_x);
	    float right_x = std::max(a_x, b_x);

	    return left_x * min_score + right_x * max_score;
	}

	Polygon polys_merger_by_width(Polygon a, Polygon b, float img_width, float long_merge_threshold){
	    float a_width = a.poly[1].X - a.poly[0].X;
	    float b_width = b.poly[1].X - b.poly[0].X;

	    float threshold_width = img_width * long_merge_threshold;

	    Polygon p;
		auto &poly = p.poly;
		poly.resize(4);
		p.score = a.score + b.score;

        auto score_inv = 1.0d / std::max(1e-8d, p.score);

        if (long_merge_threshold == -1){
                poly[0].X = (a.poly[0].X * a.score + b.poly[0].X * b.score ) * score_inv;
                poly[0].Y = (a.poly[0].Y * a.score + b.poly[0].Y * b.score ) * score_inv;
                poly[1].X = (a.poly[1].X * a.score + b.poly[1].X * b.score ) * score_inv;
                poly[1].Y = (a.poly[1].Y * a.score + b.poly[1].Y * b.score ) * score_inv;
                poly[2].X = (a.poly[2].X * a.score + b.poly[2].X * b.score ) * score_inv;
                poly[2].Y = (a.poly[2].Y * a.score + b.poly[2].Y * b.score ) * score_inv;
                poly[3].X = (a.poly[3].X * a.score + b.poly[3].X * b.score ) * score_inv;
                poly[3].Y = (a.poly[3].Y * a.score + b.poly[3].Y * b.score ) * score_inv;
        }
        else{
            if ((a_width < threshold_width) || (b_width < threshold_width)){
//    //            float max_score = std::max(a.score, b.score);
//    //            poly[0].X = (a.poly[0].X * a.score + b.poly[0].X * b.score ) * score_inv;
//                poly[0].X = left_merge(a.poly[0].X, b.poly[0].X, a.score, b.score) * score_inv;
//                poly[0].Y = (a.poly[0].Y * a.score + b.poly[0].Y * b.score ) * score_inv;
//    //            poly[1].X = (a.poly[1].X * a.score + b.poly[1].X * b.score ) * score_inv;
//                poly[1].X = right_merge(a.poly[1].X, b.poly[1].X, a.score, b.score) * score_inv;
//                poly[1].Y = (a.poly[1].Y * a.score + b.poly[1].Y * b.score ) * score_inv;
//    //            poly[2].X = (a.poly[2].X * a.score + b.poly[2].X * b.score ) * score_inv;
//                poly[2].X = right_merge(a.poly[2].X, b.poly[2].X, a.score, b.score) * score_inv;
//                poly[2].Y = (a.poly[2].Y * a.score + b.poly[2].Y * b.score ) * score_inv;
//    //            poly[3].X = (a.poly[3].X * a.score + b.poly[3].X * b.score ) * score_inv;
//                poly[3].X = left_merge(a.poly[3].X, b.poly[3].X, a.score, b.score) * score_inv;
//                poly[3].Y = (a.poly[3].Y * a.score + b.poly[3].Y * b.score ) * score_inv;
                poly[0].X = (a.poly[0].X * a.score + b.poly[0].X * b.score ) * score_inv;
                poly[0].Y = (a.poly[0].Y * a.score + b.poly[0].Y * b.score ) * score_inv;
                poly[1].X = (a.poly[1].X * a.score + b.poly[1].X * b.score ) * score_inv;
                poly[1].Y = (a.poly[1].Y * a.score + b.poly[1].Y * b.score ) * score_inv;
                poly[2].X = (a.poly[2].X * a.score + b.poly[2].X * b.score ) * score_inv;
                poly[2].Y = (a.poly[2].Y * a.score + b.poly[2].Y * b.score ) * score_inv;
                poly[3].X = (a.poly[3].X * a.score + b.poly[3].X * b.score ) * score_inv;
                poly[3].Y = (a.poly[3].Y * a.score + b.poly[3].Y * b.score ) * score_inv;
            }else{
//                  float max_score = std::max(a.score, b.score);
////    //            poly[0].X = (a.poly[0].X * a.score + b.poly[0].X * b.score ) * score_inv;
//                poly[0].X = left_merge(a.poly[0].X, b.poly[0].X, a.score, b.score) * score_inv;
//                poly[0].Y = (a.poly[0].Y * a.score + b.poly[0].Y * b.score ) * score_inv;
////    //            poly[1].X = (a.poly[1].X * a.score + b.poly[1].X * b.score ) * score_inv;
//                poly[1].X = right_merge(a.poly[1].X, b.poly[1].X, a.score, b.score) * score_inv;
//                poly[1].Y = (a.poly[1].Y * a.score + b.poly[1].Y * b.score ) * score_inv;
////    //            poly[2].X = (a.poly[2].X * a.score + b.poly[2].X * b.score ) * score_inv;
//                poly[2].X = right_merge(a.poly[2].X, b.poly[2].X, a.score, b.score) * score_inv;
//                poly[2].Y = (a.poly[2].Y * a.score + b.poly[2].Y * b.score ) * score_inv;
////    //            poly[3].X = (a.poly[3].X * a.score + b.poly[3].X * b.score ) * score_inv;
//                poly[3].X = left_merge(a.poly[3].X, b.poly[3].X, a.score, b.score) * score_inv;
//                poly[3].Y = (a.poly[3].Y * a.score + b.poly[3].Y * b.score ) * score_inv;

                poly[0].X = std::min(a.poly[0].X, b.poly[0].X);
                poly[0].Y = (a.poly[0].Y * a.score + b.poly[0].Y * b.score ) * score_inv;
                poly[1].X = std::max(a.poly[1].X, b.poly[1].X);
                poly[1].Y = (a.poly[1].Y * a.score + b.poly[1].Y * b.score ) * score_inv;
                poly[2].X = std::max(a.poly[2].X, b.poly[2].X);
                poly[2].Y = (a.poly[2].Y * a.score + b.poly[2].Y * b.score ) * score_inv;
                poly[3].X = std::min(a.poly[3].X, b.poly[3].X);
                poly[3].Y = (a.poly[3].Y * a.score + b.poly[3].Y * b.score ) * score_inv;

            }
        }
        return p;
	}





	/**
	 * The standard NMS algorithm.
	 */
	std::vector<Polygon> standard_nms(std::vector<Polygon> &polys, float iou_threshold) {
		size_t n = polys.size();
		if (n == 0)
			return {};
		std::vector<size_t> indices(n);
		std::iota(std::begin(indices), std::end(indices), 0);
		std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) { return polys[i].score > polys[j].score; });

		std::vector<size_t> keep;
		while (indices.size()) {
			size_t p = 0, cur = indices[0];
			keep.emplace_back(cur);
			for (size_t i = 1; i < indices.size(); i ++) {
				if (!should_merge(polys[cur], polys[indices[i]], iou_threshold)) {
					indices[p ++] = indices[i];
				}
			}
			indices.resize(p);
		}

		std::vector<Polygon> ret;
		for (auto &&i: keep) {
			ret.emplace_back(polys[i]);
		}
		return ret;
	}

	std::vector<Polygon>
		merge_quadrangle_n9(const double *data, size_t n, float lanms_threshold, float iou_threshold, float img_width,
		float long_merge_threshold) {
			using cInt = cl::cInt;

			// first pass
			std::vector<Polygon> polys;
			for (size_t i = 0; i < n; i ++) {
				auto p = data + i * 9;
				Polygon poly{
					{
						{cInt(p[0]), cInt(p[1])},
						{cInt(p[2]), cInt(p[3])},
						{cInt(p[4]), cInt(p[5])},
						{cInt(p[6]), cInt(p[7])},
					},
					p[8],
				};

				if (polys.size()) {
					// merge with the last one
					auto &bpoly = polys.back();
					if (should_merge(poly, bpoly, lanms_threshold)) {
						bpoly = polys_merger_by_width(poly, bpoly, img_width, long_merge_threshold);
					} else {
						polys.emplace_back(poly);
					}
				} else {
					polys.emplace_back(poly);
				}
			}
			return standard_nms(polys, iou_threshold);
		}
}
