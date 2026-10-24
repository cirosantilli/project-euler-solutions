import ProjectEulerSolutions.Lib.PrimePairCache.SmallPrefix
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchData
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows000
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows001
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows002
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows003
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows004
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows005
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows006
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows007
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows008
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows009
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows010
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows011
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows012
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows013
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows014
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows015
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows016
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows017
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows018
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows019
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows020
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows021
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows022
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows023
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows024
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows025
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows026
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows027
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows028
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows029
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows030
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows031
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows032
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows033
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows034
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows035
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows036
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows037
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows038
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows039
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows040
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows041
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows042
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows043
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows044
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows045
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows046
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows047
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows048
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows049
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows050
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows051
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows052
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows053
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows054
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows055
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows056
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows057
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows058
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows059
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows060
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows061
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows062
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows063
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows064
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows065
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows066
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows067
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows068
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows069
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows070
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows071
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows072
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows073
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows074
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows075
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows076
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows077
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows078
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows079
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows080
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows081
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows082

namespace ProjectEulerSolutions.PrimePairUpper
open PrimePairCertificate

def smallCached : Graph := PrimePairCache.smallPrefix
theorem smallCached_covers : Graph.Covers PrimePairVertices.smallTree smallCached :=
  Graph.Agrees.covers PrimePairCache.smallPrefix_agrees
def smallEmpty : Graph := .empty
theorem smallEmpty_covers : Graph.Covers PrimePairVertices.smallTree smallEmpty := Graph.empty_covers _

def smallNode2671 : Graph := .node 2671 smallRow2671 smallCached smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2671_covers : Graph.Covers PrimePairVertices.smallTree smallNode2671 :=
  Graph.node_covers _ 2671 smallRow2671 smallCached smallEmpty
    smallCached_covers smallEmpty_covers (by decide) (by decide) smallRow2671_covers

def smallNode2677 : Graph := .node 2677 smallRow2677 smallNode2671 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2677_covers : Graph.Covers PrimePairVertices.smallTree smallNode2677 :=
  Graph.node_covers _ 2677 smallRow2677 smallNode2671 smallEmpty
    smallNode2671_covers smallEmpty_covers (by decide) (by decide) smallRow2677_covers

def smallNode2687 : Graph := .node 2687 smallRow2687 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2687_covers : Graph.Covers PrimePairVertices.smallTree smallNode2687 :=
  Graph.node_covers _ 2687 smallRow2687 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2687_covers

def smallNode2689 : Graph := .node 2689 smallRow2689 smallNode2687 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2689_covers : Graph.Covers PrimePairVertices.smallTree smallNode2689 :=
  Graph.node_covers _ 2689 smallRow2689 smallNode2687 smallEmpty
    smallNode2687_covers smallEmpty_covers (by decide) (by decide) smallRow2689_covers

def smallNode2683 : Graph := .node 2683 smallRow2683 smallNode2677 smallNode2689
set_option maxRecDepth 4096 in
theorem smallNode2683_covers : Graph.Covers PrimePairVertices.smallTree smallNode2683 :=
  Graph.node_covers _ 2683 smallRow2683 smallNode2677 smallNode2689
    smallNode2677_covers smallNode2689_covers (by decide) (by decide) smallRow2683_covers

def smallNode2699 : Graph := .node 2699 smallRow2699 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2699_covers : Graph.Covers PrimePairVertices.smallTree smallNode2699 :=
  Graph.node_covers _ 2699 smallRow2699 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2699_covers

def smallNode2707 : Graph := .node 2707 smallRow2707 smallNode2699 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2707_covers : Graph.Covers PrimePairVertices.smallTree smallNode2707 :=
  Graph.node_covers _ 2707 smallRow2707 smallNode2699 smallEmpty
    smallNode2699_covers smallEmpty_covers (by decide) (by decide) smallRow2707_covers

def smallNode2713 : Graph := .node 2713 smallRow2713 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2713_covers : Graph.Covers PrimePairVertices.smallTree smallNode2713 :=
  Graph.node_covers _ 2713 smallRow2713 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2713_covers

def smallNode2711 : Graph := .node 2711 smallRow2711 smallNode2707 smallNode2713
set_option maxRecDepth 4096 in
theorem smallNode2711_covers : Graph.Covers PrimePairVertices.smallTree smallNode2711 :=
  Graph.node_covers _ 2711 smallRow2711 smallNode2707 smallNode2713
    smallNode2707_covers smallNode2713_covers (by decide) (by decide) smallRow2711_covers

def smallNode2693 : Graph := .node 2693 smallRow2693 smallNode2683 smallNode2711
set_option maxRecDepth 4096 in
theorem smallNode2693_covers : Graph.Covers PrimePairVertices.smallTree smallNode2693 :=
  Graph.node_covers _ 2693 smallRow2693 smallNode2683 smallNode2711
    smallNode2683_covers smallNode2711_covers (by decide) (by decide) smallRow2693_covers

def smallNode2729 : Graph := .node 2729 smallRow2729 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2729_covers : Graph.Covers PrimePairVertices.smallTree smallNode2729 :=
  Graph.node_covers _ 2729 smallRow2729 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2729_covers

def smallNode2731 : Graph := .node 2731 smallRow2731 smallNode2729 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2731_covers : Graph.Covers PrimePairVertices.smallTree smallNode2731 :=
  Graph.node_covers _ 2731 smallRow2731 smallNode2729 smallEmpty
    smallNode2729_covers smallEmpty_covers (by decide) (by decide) smallRow2731_covers

def smallNode2749 : Graph := .node 2749 smallRow2749 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2749_covers : Graph.Covers PrimePairVertices.smallTree smallNode2749 :=
  Graph.node_covers _ 2749 smallRow2749 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2749_covers

def smallNode2741 : Graph := .node 2741 smallRow2741 smallNode2731 smallNode2749
set_option maxRecDepth 4096 in
theorem smallNode2741_covers : Graph.Covers PrimePairVertices.smallTree smallNode2741 :=
  Graph.node_covers _ 2741 smallRow2741 smallNode2731 smallNode2749
    smallNode2731_covers smallNode2749_covers (by decide) (by decide) smallRow2741_covers

def smallNode2767 : Graph := .node 2767 smallRow2767 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2767_covers : Graph.Covers PrimePairVertices.smallTree smallNode2767 :=
  Graph.node_covers _ 2767 smallRow2767 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2767_covers

def smallNode2777 : Graph := .node 2777 smallRow2777 smallNode2767 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2777_covers : Graph.Covers PrimePairVertices.smallTree smallNode2777 :=
  Graph.node_covers _ 2777 smallRow2777 smallNode2767 smallEmpty
    smallNode2767_covers smallEmpty_covers (by decide) (by decide) smallRow2777_covers

def smallNode2791 : Graph := .node 2791 smallRow2791 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2791_covers : Graph.Covers PrimePairVertices.smallTree smallNode2791 :=
  Graph.node_covers _ 2791 smallRow2791 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2791_covers

def smallNode2789 : Graph := .node 2789 smallRow2789 smallNode2777 smallNode2791
set_option maxRecDepth 4096 in
theorem smallNode2789_covers : Graph.Covers PrimePairVertices.smallTree smallNode2789 :=
  Graph.node_covers _ 2789 smallRow2789 smallNode2777 smallNode2791
    smallNode2777_covers smallNode2791_covers (by decide) (by decide) smallRow2789_covers

def smallNode2753 : Graph := .node 2753 smallRow2753 smallNode2741 smallNode2789
set_option maxRecDepth 4096 in
theorem smallNode2753_covers : Graph.Covers PrimePairVertices.smallTree smallNode2753 :=
  Graph.node_covers _ 2753 smallRow2753 smallNode2741 smallNode2789
    smallNode2741_covers smallNode2789_covers (by decide) (by decide) smallRow2753_covers

def smallNode2719 : Graph := .node 2719 smallRow2719 smallNode2693 smallNode2753
set_option maxRecDepth 4096 in
theorem smallNode2719_covers : Graph.Covers PrimePairVertices.smallTree smallNode2719 :=
  Graph.node_covers _ 2719 smallRow2719 smallNode2693 smallNode2753
    smallNode2693_covers smallNode2753_covers (by decide) (by decide) smallRow2719_covers

def smallNode2801 : Graph := .node 2801 smallRow2801 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2801_covers : Graph.Covers PrimePairVertices.smallTree smallNode2801 :=
  Graph.node_covers _ 2801 smallRow2801 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2801_covers

def smallNode2803 : Graph := .node 2803 smallRow2803 smallNode2801 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2803_covers : Graph.Covers PrimePairVertices.smallTree smallNode2803 :=
  Graph.node_covers _ 2803 smallRow2803 smallNode2801 smallEmpty
    smallNode2801_covers smallEmpty_covers (by decide) (by decide) smallRow2803_covers

def smallNode2833 : Graph := .node 2833 smallRow2833 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2833_covers : Graph.Covers PrimePairVertices.smallTree smallNode2833 :=
  Graph.node_covers _ 2833 smallRow2833 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2833_covers

def smallNode2837 : Graph := .node 2837 smallRow2837 smallNode2833 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2837_covers : Graph.Covers PrimePairVertices.smallTree smallNode2837 :=
  Graph.node_covers _ 2837 smallRow2837 smallNode2833 smallEmpty
    smallNode2833_covers smallEmpty_covers (by decide) (by decide) smallRow2837_covers

def smallNode2819 : Graph := .node 2819 smallRow2819 smallNode2803 smallNode2837
set_option maxRecDepth 4096 in
theorem smallNode2819_covers : Graph.Covers PrimePairVertices.smallTree smallNode2819 :=
  Graph.node_covers _ 2819 smallRow2819 smallNode2803 smallNode2837
    smallNode2803_covers smallNode2837_covers (by decide) (by decide) smallRow2819_covers

def smallNode2851 : Graph := .node 2851 smallRow2851 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2851_covers : Graph.Covers PrimePairVertices.smallTree smallNode2851 :=
  Graph.node_covers _ 2851 smallRow2851 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2851_covers

def smallNode2857 : Graph := .node 2857 smallRow2857 smallNode2851 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2857_covers : Graph.Covers PrimePairVertices.smallTree smallNode2857 :=
  Graph.node_covers _ 2857 smallRow2857 smallNode2851 smallEmpty
    smallNode2851_covers smallEmpty_covers (by decide) (by decide) smallRow2857_covers

def smallNode2879 : Graph := .node 2879 smallRow2879 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2879_covers : Graph.Covers PrimePairVertices.smallTree smallNode2879 :=
  Graph.node_covers _ 2879 smallRow2879 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2879_covers

def smallNode2861 : Graph := .node 2861 smallRow2861 smallNode2857 smallNode2879
set_option maxRecDepth 4096 in
theorem smallNode2861_covers : Graph.Covers PrimePairVertices.smallTree smallNode2861 :=
  Graph.node_covers _ 2861 smallRow2861 smallNode2857 smallNode2879
    smallNode2857_covers smallNode2879_covers (by decide) (by decide) smallRow2861_covers

def smallNode2843 : Graph := .node 2843 smallRow2843 smallNode2819 smallNode2861
set_option maxRecDepth 4096 in
theorem smallNode2843_covers : Graph.Covers PrimePairVertices.smallTree smallNode2843 :=
  Graph.node_covers _ 2843 smallRow2843 smallNode2819 smallNode2861
    smallNode2819_covers smallNode2861_covers (by decide) (by decide) smallRow2843_covers

def smallNode2897 : Graph := .node 2897 smallRow2897 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2897_covers : Graph.Covers PrimePairVertices.smallTree smallNode2897 :=
  Graph.node_covers _ 2897 smallRow2897 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2897_covers

def smallNode2903 : Graph := .node 2903 smallRow2903 smallNode2897 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2903_covers : Graph.Covers PrimePairVertices.smallTree smallNode2903 :=
  Graph.node_covers _ 2903 smallRow2903 smallNode2897 smallEmpty
    smallNode2897_covers smallEmpty_covers (by decide) (by decide) smallRow2903_covers

def smallNode2917 : Graph := .node 2917 smallRow2917 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2917_covers : Graph.Covers PrimePairVertices.smallTree smallNode2917 :=
  Graph.node_covers _ 2917 smallRow2917 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2917_covers

def smallNode2909 : Graph := .node 2909 smallRow2909 smallNode2903 smallNode2917
set_option maxRecDepth 4096 in
theorem smallNode2909_covers : Graph.Covers PrimePairVertices.smallTree smallNode2909 :=
  Graph.node_covers _ 2909 smallRow2909 smallNode2903 smallNode2917
    smallNode2903_covers smallNode2917_covers (by decide) (by decide) smallRow2909_covers

def smallNode2939 : Graph := .node 2939 smallRow2939 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2939_covers : Graph.Covers PrimePairVertices.smallTree smallNode2939 :=
  Graph.node_covers _ 2939 smallRow2939 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2939_covers

def smallNode2953 : Graph := .node 2953 smallRow2953 smallNode2939 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2953_covers : Graph.Covers PrimePairVertices.smallTree smallNode2953 :=
  Graph.node_covers _ 2953 smallRow2953 smallNode2939 smallEmpty
    smallNode2939_covers smallEmpty_covers (by decide) (by decide) smallRow2953_covers

def smallNode2963 : Graph := .node 2963 smallRow2963 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2963_covers : Graph.Covers PrimePairVertices.smallTree smallNode2963 :=
  Graph.node_covers _ 2963 smallRow2963 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2963_covers

def smallNode2957 : Graph := .node 2957 smallRow2957 smallNode2953 smallNode2963
set_option maxRecDepth 4096 in
theorem smallNode2957_covers : Graph.Covers PrimePairVertices.smallTree smallNode2957 :=
  Graph.node_covers _ 2957 smallRow2957 smallNode2953 smallNode2963
    smallNode2953_covers smallNode2963_covers (by decide) (by decide) smallRow2957_covers

def smallNode2927 : Graph := .node 2927 smallRow2927 smallNode2909 smallNode2957
set_option maxRecDepth 4096 in
theorem smallNode2927_covers : Graph.Covers PrimePairVertices.smallTree smallNode2927 :=
  Graph.node_covers _ 2927 smallRow2927 smallNode2909 smallNode2957
    smallNode2909_covers smallNode2957_covers (by decide) (by decide) smallRow2927_covers

def smallNode2887 : Graph := .node 2887 smallRow2887 smallNode2843 smallNode2927
set_option maxRecDepth 4096 in
theorem smallNode2887_covers : Graph.Covers PrimePairVertices.smallTree smallNode2887 :=
  Graph.node_covers _ 2887 smallRow2887 smallNode2843 smallNode2927
    smallNode2843_covers smallNode2927_covers (by decide) (by decide) smallRow2887_covers

def smallNode2797 : Graph := .node 2797 smallRow2797 smallNode2719 smallNode2887
set_option maxRecDepth 4096 in
theorem smallNode2797_covers : Graph.Covers PrimePairVertices.smallTree smallNode2797 :=
  Graph.node_covers _ 2797 smallRow2797 smallNode2719 smallNode2887
    smallNode2719_covers smallNode2887_covers (by decide) (by decide) smallRow2797_covers

def smallNode2971 : Graph := .node 2971 smallRow2971 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2971_covers : Graph.Covers PrimePairVertices.smallTree smallNode2971 :=
  Graph.node_covers _ 2971 smallRow2971 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow2971_covers

def smallNode2999 : Graph := .node 2999 smallRow2999 smallNode2971 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode2999_covers : Graph.Covers PrimePairVertices.smallTree smallNode2999 :=
  Graph.node_covers _ 2999 smallRow2999 smallNode2971 smallEmpty
    smallNode2971_covers smallEmpty_covers (by decide) (by decide) smallRow2999_covers

def smallNode3011 : Graph := .node 3011 smallRow3011 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3011_covers : Graph.Covers PrimePairVertices.smallTree smallNode3011 :=
  Graph.node_covers _ 3011 smallRow3011 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3011_covers

def smallNode3019 : Graph := .node 3019 smallRow3019 smallNode3011 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3019_covers : Graph.Covers PrimePairVertices.smallTree smallNode3019 :=
  Graph.node_covers _ 3019 smallRow3019 smallNode3011 smallEmpty
    smallNode3011_covers smallEmpty_covers (by decide) (by decide) smallRow3019_covers

def smallNode3001 : Graph := .node 3001 smallRow3001 smallNode2999 smallNode3019
set_option maxRecDepth 4096 in
theorem smallNode3001_covers : Graph.Covers PrimePairVertices.smallTree smallNode3001 :=
  Graph.node_covers _ 3001 smallRow3001 smallNode2999 smallNode3019
    smallNode2999_covers smallNode3019_covers (by decide) (by decide) smallRow3001_covers

def smallNode3037 : Graph := .node 3037 smallRow3037 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3037_covers : Graph.Covers PrimePairVertices.smallTree smallNode3037 :=
  Graph.node_covers _ 3037 smallRow3037 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3037_covers

def smallNode3041 : Graph := .node 3041 smallRow3041 smallNode3037 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3041_covers : Graph.Covers PrimePairVertices.smallTree smallNode3041 :=
  Graph.node_covers _ 3041 smallRow3041 smallNode3037 smallEmpty
    smallNode3037_covers smallEmpty_covers (by decide) (by decide) smallRow3041_covers

def smallNode3061 : Graph := .node 3061 smallRow3061 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3061_covers : Graph.Covers PrimePairVertices.smallTree smallNode3061 :=
  Graph.node_covers _ 3061 smallRow3061 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3061_covers

def smallNode3049 : Graph := .node 3049 smallRow3049 smallNode3041 smallNode3061
set_option maxRecDepth 4096 in
theorem smallNode3049_covers : Graph.Covers PrimePairVertices.smallTree smallNode3049 :=
  Graph.node_covers _ 3049 smallRow3049 smallNode3041 smallNode3061
    smallNode3041_covers smallNode3061_covers (by decide) (by decide) smallRow3049_covers

def smallNode3023 : Graph := .node 3023 smallRow3023 smallNode3001 smallNode3049
set_option maxRecDepth 4096 in
theorem smallNode3023_covers : Graph.Covers PrimePairVertices.smallTree smallNode3023 :=
  Graph.node_covers _ 3023 smallRow3023 smallNode3001 smallNode3049
    smallNode3001_covers smallNode3049_covers (by decide) (by decide) smallRow3023_covers

def smallNode3079 : Graph := .node 3079 smallRow3079 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3079_covers : Graph.Covers PrimePairVertices.smallTree smallNode3079 :=
  Graph.node_covers _ 3079 smallRow3079 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3079_covers

def smallNode3083 : Graph := .node 3083 smallRow3083 smallNode3079 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3083_covers : Graph.Covers PrimePairVertices.smallTree smallNode3083 :=
  Graph.node_covers _ 3083 smallRow3083 smallNode3079 smallEmpty
    smallNode3079_covers smallEmpty_covers (by decide) (by decide) smallRow3083_covers

def smallNode3109 : Graph := .node 3109 smallRow3109 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3109_covers : Graph.Covers PrimePairVertices.smallTree smallNode3109 :=
  Graph.node_covers _ 3109 smallRow3109 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3109_covers

def smallNode3089 : Graph := .node 3089 smallRow3089 smallNode3083 smallNode3109
set_option maxRecDepth 4096 in
theorem smallNode3089_covers : Graph.Covers PrimePairVertices.smallTree smallNode3089 :=
  Graph.node_covers _ 3089 smallRow3089 smallNode3083 smallNode3109
    smallNode3083_covers smallNode3109_covers (by decide) (by decide) smallRow3089_covers

def smallNode3121 : Graph := .node 3121 smallRow3121 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3121_covers : Graph.Covers PrimePairVertices.smallTree smallNode3121 :=
  Graph.node_covers _ 3121 smallRow3121 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3121_covers

def smallNode3137 : Graph := .node 3137 smallRow3137 smallNode3121 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3137_covers : Graph.Covers PrimePairVertices.smallTree smallNode3137 :=
  Graph.node_covers _ 3137 smallRow3137 smallNode3121 smallEmpty
    smallNode3121_covers smallEmpty_covers (by decide) (by decide) smallRow3137_covers

def smallNode3167 : Graph := .node 3167 smallRow3167 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3167_covers : Graph.Covers PrimePairVertices.smallTree smallNode3167 :=
  Graph.node_covers _ 3167 smallRow3167 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3167_covers

def smallNode3163 : Graph := .node 3163 smallRow3163 smallNode3137 smallNode3167
set_option maxRecDepth 4096 in
theorem smallNode3163_covers : Graph.Covers PrimePairVertices.smallTree smallNode3163 :=
  Graph.node_covers _ 3163 smallRow3163 smallNode3137 smallNode3167
    smallNode3137_covers smallNode3167_covers (by decide) (by decide) smallRow3163_covers

def smallNode3119 : Graph := .node 3119 smallRow3119 smallNode3089 smallNode3163
set_option maxRecDepth 4096 in
theorem smallNode3119_covers : Graph.Covers PrimePairVertices.smallTree smallNode3119 :=
  Graph.node_covers _ 3119 smallRow3119 smallNode3089 smallNode3163
    smallNode3089_covers smallNode3163_covers (by decide) (by decide) smallRow3119_covers

def smallNode3067 : Graph := .node 3067 smallRow3067 smallNode3023 smallNode3119
set_option maxRecDepth 4096 in
theorem smallNode3067_covers : Graph.Covers PrimePairVertices.smallTree smallNode3067 :=
  Graph.node_covers _ 3067 smallRow3067 smallNode3023 smallNode3119
    smallNode3023_covers smallNode3119_covers (by decide) (by decide) smallRow3067_covers

def smallNode3181 : Graph := .node 3181 smallRow3181 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3181_covers : Graph.Covers PrimePairVertices.smallTree smallNode3181 :=
  Graph.node_covers _ 3181 smallRow3181 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3181_covers

def smallNode3187 : Graph := .node 3187 smallRow3187 smallNode3181 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3187_covers : Graph.Covers PrimePairVertices.smallTree smallNode3187 :=
  Graph.node_covers _ 3187 smallRow3187 smallNode3181 smallEmpty
    smallNode3181_covers smallEmpty_covers (by decide) (by decide) smallRow3187_covers

def smallNode3203 : Graph := .node 3203 smallRow3203 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3203_covers : Graph.Covers PrimePairVertices.smallTree smallNode3203 :=
  Graph.node_covers _ 3203 smallRow3203 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3203_covers

def smallNode3209 : Graph := .node 3209 smallRow3209 smallNode3203 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3209_covers : Graph.Covers PrimePairVertices.smallTree smallNode3209 :=
  Graph.node_covers _ 3209 smallRow3209 smallNode3203 smallEmpty
    smallNode3203_covers smallEmpty_covers (by decide) (by decide) smallRow3209_covers

def smallNode3191 : Graph := .node 3191 smallRow3191 smallNode3187 smallNode3209
set_option maxRecDepth 4096 in
theorem smallNode3191_covers : Graph.Covers PrimePairVertices.smallTree smallNode3191 :=
  Graph.node_covers _ 3191 smallRow3191 smallNode3187 smallNode3209
    smallNode3187_covers smallNode3209_covers (by decide) (by decide) smallRow3191_covers

def smallNode3221 : Graph := .node 3221 smallRow3221 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3221_covers : Graph.Covers PrimePairVertices.smallTree smallNode3221 :=
  Graph.node_covers _ 3221 smallRow3221 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3221_covers

def smallNode3229 : Graph := .node 3229 smallRow3229 smallNode3221 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3229_covers : Graph.Covers PrimePairVertices.smallTree smallNode3229 :=
  Graph.node_covers _ 3229 smallRow3229 smallNode3221 smallEmpty
    smallNode3221_covers smallEmpty_covers (by decide) (by decide) smallRow3229_covers

def smallNode3253 : Graph := .node 3253 smallRow3253 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3253_covers : Graph.Covers PrimePairVertices.smallTree smallNode3253 :=
  Graph.node_covers _ 3253 smallRow3253 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3253_covers

def smallNode3251 : Graph := .node 3251 smallRow3251 smallNode3229 smallNode3253
set_option maxRecDepth 4096 in
theorem smallNode3251_covers : Graph.Covers PrimePairVertices.smallTree smallNode3251 :=
  Graph.node_covers _ 3251 smallRow3251 smallNode3229 smallNode3253
    smallNode3229_covers smallNode3253_covers (by decide) (by decide) smallRow3251_covers

def smallNode3217 : Graph := .node 3217 smallRow3217 smallNode3191 smallNode3251
set_option maxRecDepth 4096 in
theorem smallNode3217_covers : Graph.Covers PrimePairVertices.smallTree smallNode3217 :=
  Graph.node_covers _ 3217 smallRow3217 smallNode3191 smallNode3251
    smallNode3191_covers smallNode3251_covers (by decide) (by decide) smallRow3217_covers

def smallNode3259 : Graph := .node 3259 smallRow3259 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3259_covers : Graph.Covers PrimePairVertices.smallTree smallNode3259 :=
  Graph.node_covers _ 3259 smallRow3259 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3259_covers

def smallNode3271 : Graph := .node 3271 smallRow3271 smallNode3259 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3271_covers : Graph.Covers PrimePairVertices.smallTree smallNode3271 :=
  Graph.node_covers _ 3271 smallRow3271 smallNode3259 smallEmpty
    smallNode3259_covers smallEmpty_covers (by decide) (by decide) smallRow3271_covers

def smallNode3301 : Graph := .node 3301 smallRow3301 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3301_covers : Graph.Covers PrimePairVertices.smallTree smallNode3301 :=
  Graph.node_covers _ 3301 smallRow3301 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3301_covers

def smallNode3299 : Graph := .node 3299 smallRow3299 smallNode3271 smallNode3301
set_option maxRecDepth 4096 in
theorem smallNode3299_covers : Graph.Covers PrimePairVertices.smallTree smallNode3299 :=
  Graph.node_covers _ 3299 smallRow3299 smallNode3271 smallNode3301
    smallNode3271_covers smallNode3301_covers (by decide) (by decide) smallRow3299_covers

def smallNode3313 : Graph := .node 3313 smallRow3313 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3313_covers : Graph.Covers PrimePairVertices.smallTree smallNode3313 :=
  Graph.node_covers _ 3313 smallRow3313 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3313_covers

def smallNode3319 : Graph := .node 3319 smallRow3319 smallNode3313 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3319_covers : Graph.Covers PrimePairVertices.smallTree smallNode3319 :=
  Graph.node_covers _ 3319 smallRow3319 smallNode3313 smallEmpty
    smallNode3313_covers smallEmpty_covers (by decide) (by decide) smallRow3319_covers

def smallNode3329 : Graph := .node 3329 smallRow3329 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3329_covers : Graph.Covers PrimePairVertices.smallTree smallNode3329 :=
  Graph.node_covers _ 3329 smallRow3329 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3329_covers

def smallNode3323 : Graph := .node 3323 smallRow3323 smallNode3319 smallNode3329
set_option maxRecDepth 4096 in
theorem smallNode3323_covers : Graph.Covers PrimePairVertices.smallTree smallNode3323 :=
  Graph.node_covers _ 3323 smallRow3323 smallNode3319 smallNode3329
    smallNode3319_covers smallNode3329_covers (by decide) (by decide) smallRow3323_covers

def smallNode3307 : Graph := .node 3307 smallRow3307 smallNode3299 smallNode3323
set_option maxRecDepth 4096 in
theorem smallNode3307_covers : Graph.Covers PrimePairVertices.smallTree smallNode3307 :=
  Graph.node_covers _ 3307 smallRow3307 smallNode3299 smallNode3323
    smallNode3299_covers smallNode3323_covers (by decide) (by decide) smallRow3307_covers

def smallNode3257 : Graph := .node 3257 smallRow3257 smallNode3217 smallNode3307
set_option maxRecDepth 4096 in
theorem smallNode3257_covers : Graph.Covers PrimePairVertices.smallTree smallNode3257 :=
  Graph.node_covers _ 3257 smallRow3257 smallNode3217 smallNode3307
    smallNode3217_covers smallNode3307_covers (by decide) (by decide) smallRow3257_covers

def smallNode3169 : Graph := .node 3169 smallRow3169 smallNode3067 smallNode3257
set_option maxRecDepth 4096 in
theorem smallNode3169_covers : Graph.Covers PrimePairVertices.smallTree smallNode3169 :=
  Graph.node_covers _ 3169 smallRow3169 smallNode3067 smallNode3257
    smallNode3067_covers smallNode3257_covers (by decide) (by decide) smallRow3169_covers

def smallNode2969 : Graph := .node 2969 smallRow2969 smallNode2797 smallNode3169
set_option maxRecDepth 4096 in
theorem smallNode2969_covers : Graph.Covers PrimePairVertices.smallTree smallNode2969 :=
  Graph.node_covers _ 2969 smallRow2969 smallNode2797 smallNode3169
    smallNode2797_covers smallNode3169_covers (by decide) (by decide) smallRow2969_covers

def smallNode3343 : Graph := .node 3343 smallRow3343 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3343_covers : Graph.Covers PrimePairVertices.smallTree smallNode3343 :=
  Graph.node_covers _ 3343 smallRow3343 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3343_covers

def smallNode3347 : Graph := .node 3347 smallRow3347 smallNode3343 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3347_covers : Graph.Covers PrimePairVertices.smallTree smallNode3347 :=
  Graph.node_covers _ 3347 smallRow3347 smallNode3343 smallEmpty
    smallNode3343_covers smallEmpty_covers (by decide) (by decide) smallRow3347_covers

def smallNode3361 : Graph := .node 3361 smallRow3361 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3361_covers : Graph.Covers PrimePairVertices.smallTree smallNode3361 :=
  Graph.node_covers _ 3361 smallRow3361 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3361_covers

def smallNode3371 : Graph := .node 3371 smallRow3371 smallNode3361 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3371_covers : Graph.Covers PrimePairVertices.smallTree smallNode3371 :=
  Graph.node_covers _ 3371 smallRow3371 smallNode3361 smallEmpty
    smallNode3361_covers smallEmpty_covers (by decide) (by decide) smallRow3371_covers

def smallNode3359 : Graph := .node 3359 smallRow3359 smallNode3347 smallNode3371
set_option maxRecDepth 4096 in
theorem smallNode3359_covers : Graph.Covers PrimePairVertices.smallTree smallNode3359 :=
  Graph.node_covers _ 3359 smallRow3359 smallNode3347 smallNode3371
    smallNode3347_covers smallNode3371_covers (by decide) (by decide) smallRow3359_covers

def smallNode3389 : Graph := .node 3389 smallRow3389 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3389_covers : Graph.Covers PrimePairVertices.smallTree smallNode3389 :=
  Graph.node_covers _ 3389 smallRow3389 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3389_covers

def smallNode3391 : Graph := .node 3391 smallRow3391 smallNode3389 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3391_covers : Graph.Covers PrimePairVertices.smallTree smallNode3391 :=
  Graph.node_covers _ 3391 smallRow3391 smallNode3389 smallEmpty
    smallNode3389_covers smallEmpty_covers (by decide) (by decide) smallRow3391_covers

def smallNode3413 : Graph := .node 3413 smallRow3413 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3413_covers : Graph.Covers PrimePairVertices.smallTree smallNode3413 :=
  Graph.node_covers _ 3413 smallRow3413 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3413_covers

def smallNode3407 : Graph := .node 3407 smallRow3407 smallNode3391 smallNode3413
set_option maxRecDepth 4096 in
theorem smallNode3407_covers : Graph.Covers PrimePairVertices.smallTree smallNode3407 :=
  Graph.node_covers _ 3407 smallRow3407 smallNode3391 smallNode3413
    smallNode3391_covers smallNode3413_covers (by decide) (by decide) smallRow3407_covers

def smallNode3373 : Graph := .node 3373 smallRow3373 smallNode3359 smallNode3407
set_option maxRecDepth 4096 in
theorem smallNode3373_covers : Graph.Covers PrimePairVertices.smallTree smallNode3373 :=
  Graph.node_covers _ 3373 smallRow3373 smallNode3359 smallNode3407
    smallNode3359_covers smallNode3407_covers (by decide) (by decide) smallRow3373_covers

def smallNode3449 : Graph := .node 3449 smallRow3449 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3449_covers : Graph.Covers PrimePairVertices.smallTree smallNode3449 :=
  Graph.node_covers _ 3449 smallRow3449 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3449_covers

def smallNode3457 : Graph := .node 3457 smallRow3457 smallNode3449 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3457_covers : Graph.Covers PrimePairVertices.smallTree smallNode3457 :=
  Graph.node_covers _ 3457 smallRow3457 smallNode3449 smallEmpty
    smallNode3449_covers smallEmpty_covers (by decide) (by decide) smallRow3457_covers

def smallNode3463 : Graph := .node 3463 smallRow3463 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3463_covers : Graph.Covers PrimePairVertices.smallTree smallNode3463 :=
  Graph.node_covers _ 3463 smallRow3463 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3463_covers

def smallNode3461 : Graph := .node 3461 smallRow3461 smallNode3457 smallNode3463
set_option maxRecDepth 4096 in
theorem smallNode3461_covers : Graph.Covers PrimePairVertices.smallTree smallNode3461 :=
  Graph.node_covers _ 3461 smallRow3461 smallNode3457 smallNode3463
    smallNode3457_covers smallNode3463_covers (by decide) (by decide) smallRow3461_covers

def smallNode3469 : Graph := .node 3469 smallRow3469 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3469_covers : Graph.Covers PrimePairVertices.smallTree smallNode3469 :=
  Graph.node_covers _ 3469 smallRow3469 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3469_covers

def smallNode3491 : Graph := .node 3491 smallRow3491 smallNode3469 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3491_covers : Graph.Covers PrimePairVertices.smallTree smallNode3491 :=
  Graph.node_covers _ 3491 smallRow3491 smallNode3469 smallEmpty
    smallNode3469_covers smallEmpty_covers (by decide) (by decide) smallRow3491_covers

def smallNode3511 : Graph := .node 3511 smallRow3511 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3511_covers : Graph.Covers PrimePairVertices.smallTree smallNode3511 :=
  Graph.node_covers _ 3511 smallRow3511 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3511_covers

def smallNode3499 : Graph := .node 3499 smallRow3499 smallNode3491 smallNode3511
set_option maxRecDepth 4096 in
theorem smallNode3499_covers : Graph.Covers PrimePairVertices.smallTree smallNode3499 :=
  Graph.node_covers _ 3499 smallRow3499 smallNode3491 smallNode3511
    smallNode3491_covers smallNode3511_covers (by decide) (by decide) smallRow3499_covers

def smallNode3467 : Graph := .node 3467 smallRow3467 smallNode3461 smallNode3499
set_option maxRecDepth 4096 in
theorem smallNode3467_covers : Graph.Covers PrimePairVertices.smallTree smallNode3467 :=
  Graph.node_covers _ 3467 smallRow3467 smallNode3461 smallNode3499
    smallNode3461_covers smallNode3499_covers (by decide) (by decide) smallRow3467_covers

def smallNode3433 : Graph := .node 3433 smallRow3433 smallNode3373 smallNode3467
set_option maxRecDepth 4096 in
theorem smallNode3433_covers : Graph.Covers PrimePairVertices.smallTree smallNode3433 :=
  Graph.node_covers _ 3433 smallRow3433 smallNode3373 smallNode3467
    smallNode3373_covers smallNode3467_covers (by decide) (by decide) smallRow3433_covers

def smallNode3527 : Graph := .node 3527 smallRow3527 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3527_covers : Graph.Covers PrimePairVertices.smallTree smallNode3527 :=
  Graph.node_covers _ 3527 smallRow3527 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3527_covers

def smallNode3529 : Graph := .node 3529 smallRow3529 smallNode3527 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3529_covers : Graph.Covers PrimePairVertices.smallTree smallNode3529 :=
  Graph.node_covers _ 3529 smallRow3529 smallNode3527 smallEmpty
    smallNode3527_covers smallEmpty_covers (by decide) (by decide) smallRow3529_covers

def smallNode3539 : Graph := .node 3539 smallRow3539 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3539_covers : Graph.Covers PrimePairVertices.smallTree smallNode3539 :=
  Graph.node_covers _ 3539 smallRow3539 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3539_covers

def smallNode3541 : Graph := .node 3541 smallRow3541 smallNode3539 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3541_covers : Graph.Covers PrimePairVertices.smallTree smallNode3541 :=
  Graph.node_covers _ 3541 smallRow3541 smallNode3539 smallEmpty
    smallNode3539_covers smallEmpty_covers (by decide) (by decide) smallRow3541_covers

def smallNode3533 : Graph := .node 3533 smallRow3533 smallNode3529 smallNode3541
set_option maxRecDepth 4096 in
theorem smallNode3533_covers : Graph.Covers PrimePairVertices.smallTree smallNode3533 :=
  Graph.node_covers _ 3533 smallRow3533 smallNode3529 smallNode3541
    smallNode3529_covers smallNode3541_covers (by decide) (by decide) smallRow3533_covers

def smallNode3557 : Graph := .node 3557 smallRow3557 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3557_covers : Graph.Covers PrimePairVertices.smallTree smallNode3557 :=
  Graph.node_covers _ 3557 smallRow3557 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3557_covers

def smallNode3559 : Graph := .node 3559 smallRow3559 smallNode3557 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3559_covers : Graph.Covers PrimePairVertices.smallTree smallNode3559 :=
  Graph.node_covers _ 3559 smallRow3559 smallNode3557 smallEmpty
    smallNode3557_covers smallEmpty_covers (by decide) (by decide) smallRow3559_covers

def smallNode3581 : Graph := .node 3581 smallRow3581 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3581_covers : Graph.Covers PrimePairVertices.smallTree smallNode3581 :=
  Graph.node_covers _ 3581 smallRow3581 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3581_covers

def smallNode3571 : Graph := .node 3571 smallRow3571 smallNode3559 smallNode3581
set_option maxRecDepth 4096 in
theorem smallNode3571_covers : Graph.Covers PrimePairVertices.smallTree smallNode3571 :=
  Graph.node_covers _ 3571 smallRow3571 smallNode3559 smallNode3581
    smallNode3559_covers smallNode3581_covers (by decide) (by decide) smallRow3571_covers

def smallNode3547 : Graph := .node 3547 smallRow3547 smallNode3533 smallNode3571
set_option maxRecDepth 4096 in
theorem smallNode3547_covers : Graph.Covers PrimePairVertices.smallTree smallNode3547 :=
  Graph.node_covers _ 3547 smallRow3547 smallNode3533 smallNode3571
    smallNode3533_covers smallNode3571_covers (by decide) (by decide) smallRow3547_covers

def smallNode3593 : Graph := .node 3593 smallRow3593 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3593_covers : Graph.Covers PrimePairVertices.smallTree smallNode3593 :=
  Graph.node_covers _ 3593 smallRow3593 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3593_covers

def smallNode3607 : Graph := .node 3607 smallRow3607 smallNode3593 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3607_covers : Graph.Covers PrimePairVertices.smallTree smallNode3607 :=
  Graph.node_covers _ 3607 smallRow3607 smallNode3593 smallEmpty
    smallNode3593_covers smallEmpty_covers (by decide) (by decide) smallRow3607_covers

def smallNode3617 : Graph := .node 3617 smallRow3617 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3617_covers : Graph.Covers PrimePairVertices.smallTree smallNode3617 :=
  Graph.node_covers _ 3617 smallRow3617 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3617_covers

def smallNode3613 : Graph := .node 3613 smallRow3613 smallNode3607 smallNode3617
set_option maxRecDepth 4096 in
theorem smallNode3613_covers : Graph.Covers PrimePairVertices.smallTree smallNode3613 :=
  Graph.node_covers _ 3613 smallRow3613 smallNode3607 smallNode3617
    smallNode3607_covers smallNode3617_covers (by decide) (by decide) smallRow3613_covers

def smallNode3631 : Graph := .node 3631 smallRow3631 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3631_covers : Graph.Covers PrimePairVertices.smallTree smallNode3631 :=
  Graph.node_covers _ 3631 smallRow3631 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3631_covers

def smallNode3637 : Graph := .node 3637 smallRow3637 smallNode3631 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3637_covers : Graph.Covers PrimePairVertices.smallTree smallNode3637 :=
  Graph.node_covers _ 3637 smallRow3637 smallNode3631 smallEmpty
    smallNode3631_covers smallEmpty_covers (by decide) (by decide) smallRow3637_covers

def smallNode3659 : Graph := .node 3659 smallRow3659 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3659_covers : Graph.Covers PrimePairVertices.smallTree smallNode3659 :=
  Graph.node_covers _ 3659 smallRow3659 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3659_covers

def smallNode3643 : Graph := .node 3643 smallRow3643 smallNode3637 smallNode3659
set_option maxRecDepth 4096 in
theorem smallNode3643_covers : Graph.Covers PrimePairVertices.smallTree smallNode3643 :=
  Graph.node_covers _ 3643 smallRow3643 smallNode3637 smallNode3659
    smallNode3637_covers smallNode3659_covers (by decide) (by decide) smallRow3643_covers

def smallNode3623 : Graph := .node 3623 smallRow3623 smallNode3613 smallNode3643
set_option maxRecDepth 4096 in
theorem smallNode3623_covers : Graph.Covers PrimePairVertices.smallTree smallNode3623 :=
  Graph.node_covers _ 3623 smallRow3623 smallNode3613 smallNode3643
    smallNode3613_covers smallNode3643_covers (by decide) (by decide) smallRow3623_covers

def smallNode3583 : Graph := .node 3583 smallRow3583 smallNode3547 smallNode3623
set_option maxRecDepth 4096 in
theorem smallNode3583_covers : Graph.Covers PrimePairVertices.smallTree smallNode3583 :=
  Graph.node_covers _ 3583 smallRow3583 smallNode3547 smallNode3623
    smallNode3547_covers smallNode3623_covers (by decide) (by decide) smallRow3583_covers

def smallNode3517 : Graph := .node 3517 smallRow3517 smallNode3433 smallNode3583
set_option maxRecDepth 4096 in
theorem smallNode3517_covers : Graph.Covers PrimePairVertices.smallTree smallNode3517 :=
  Graph.node_covers _ 3517 smallRow3517 smallNode3433 smallNode3583
    smallNode3433_covers smallNode3583_covers (by decide) (by decide) smallRow3517_covers

def smallNode3673 : Graph := .node 3673 smallRow3673 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3673_covers : Graph.Covers PrimePairVertices.smallTree smallNode3673 :=
  Graph.node_covers _ 3673 smallRow3673 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3673_covers

def smallNode3677 : Graph := .node 3677 smallRow3677 smallNode3673 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3677_covers : Graph.Covers PrimePairVertices.smallTree smallNode3677 :=
  Graph.node_covers _ 3677 smallRow3677 smallNode3673 smallEmpty
    smallNode3673_covers smallEmpty_covers (by decide) (by decide) smallRow3677_covers

def smallNode3697 : Graph := .node 3697 smallRow3697 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3697_covers : Graph.Covers PrimePairVertices.smallTree smallNode3697 :=
  Graph.node_covers _ 3697 smallRow3697 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3697_covers

def smallNode3701 : Graph := .node 3701 smallRow3701 smallNode3697 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3701_covers : Graph.Covers PrimePairVertices.smallTree smallNode3701 :=
  Graph.node_covers _ 3701 smallRow3701 smallNode3697 smallEmpty
    smallNode3697_covers smallEmpty_covers (by decide) (by decide) smallRow3701_covers

def smallNode3691 : Graph := .node 3691 smallRow3691 smallNode3677 smallNode3701
set_option maxRecDepth 4096 in
theorem smallNode3691_covers : Graph.Covers PrimePairVertices.smallTree smallNode3691 :=
  Graph.node_covers _ 3691 smallRow3691 smallNode3677 smallNode3701
    smallNode3677_covers smallNode3701_covers (by decide) (by decide) smallRow3691_covers

def smallNode3719 : Graph := .node 3719 smallRow3719 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3719_covers : Graph.Covers PrimePairVertices.smallTree smallNode3719 :=
  Graph.node_covers _ 3719 smallRow3719 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3719_covers

def smallNode3727 : Graph := .node 3727 smallRow3727 smallNode3719 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3727_covers : Graph.Covers PrimePairVertices.smallTree smallNode3727 :=
  Graph.node_covers _ 3727 smallRow3727 smallNode3719 smallEmpty
    smallNode3719_covers smallEmpty_covers (by decide) (by decide) smallRow3727_covers

def smallNode3739 : Graph := .node 3739 smallRow3739 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3739_covers : Graph.Covers PrimePairVertices.smallTree smallNode3739 :=
  Graph.node_covers _ 3739 smallRow3739 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3739_covers

def smallNode3733 : Graph := .node 3733 smallRow3733 smallNode3727 smallNode3739
set_option maxRecDepth 4096 in
theorem smallNode3733_covers : Graph.Covers PrimePairVertices.smallTree smallNode3733 :=
  Graph.node_covers _ 3733 smallRow3733 smallNode3727 smallNode3739
    smallNode3727_covers smallNode3739_covers (by decide) (by decide) smallRow3733_covers

def smallNode3709 : Graph := .node 3709 smallRow3709 smallNode3691 smallNode3733
set_option maxRecDepth 4096 in
theorem smallNode3709_covers : Graph.Covers PrimePairVertices.smallTree smallNode3709 :=
  Graph.node_covers _ 3709 smallRow3709 smallNode3691 smallNode3733
    smallNode3691_covers smallNode3733_covers (by decide) (by decide) smallRow3709_covers

def smallNode3767 : Graph := .node 3767 smallRow3767 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3767_covers : Graph.Covers PrimePairVertices.smallTree smallNode3767 :=
  Graph.node_covers _ 3767 smallRow3767 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3767_covers

def smallNode3769 : Graph := .node 3769 smallRow3769 smallNode3767 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3769_covers : Graph.Covers PrimePairVertices.smallTree smallNode3769 :=
  Graph.node_covers _ 3769 smallRow3769 smallNode3767 smallEmpty
    smallNode3767_covers smallEmpty_covers (by decide) (by decide) smallRow3769_covers

def smallNode3793 : Graph := .node 3793 smallRow3793 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3793_covers : Graph.Covers PrimePairVertices.smallTree smallNode3793 :=
  Graph.node_covers _ 3793 smallRow3793 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3793_covers

def smallNode3779 : Graph := .node 3779 smallRow3779 smallNode3769 smallNode3793
set_option maxRecDepth 4096 in
theorem smallNode3779_covers : Graph.Covers PrimePairVertices.smallTree smallNode3779 :=
  Graph.node_covers _ 3779 smallRow3779 smallNode3769 smallNode3793
    smallNode3769_covers smallNode3793_covers (by decide) (by decide) smallRow3779_covers

def smallNode3803 : Graph := .node 3803 smallRow3803 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3803_covers : Graph.Covers PrimePairVertices.smallTree smallNode3803 :=
  Graph.node_covers _ 3803 smallRow3803 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3803_covers

def smallNode3821 : Graph := .node 3821 smallRow3821 smallNode3803 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3821_covers : Graph.Covers PrimePairVertices.smallTree smallNode3821 :=
  Graph.node_covers _ 3821 smallRow3821 smallNode3803 smallEmpty
    smallNode3803_covers smallEmpty_covers (by decide) (by decide) smallRow3821_covers

def smallNode3833 : Graph := .node 3833 smallRow3833 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3833_covers : Graph.Covers PrimePairVertices.smallTree smallNode3833 :=
  Graph.node_covers _ 3833 smallRow3833 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3833_covers

def smallNode3823 : Graph := .node 3823 smallRow3823 smallNode3821 smallNode3833
set_option maxRecDepth 4096 in
theorem smallNode3823_covers : Graph.Covers PrimePairVertices.smallTree smallNode3823 :=
  Graph.node_covers _ 3823 smallRow3823 smallNode3821 smallNode3833
    smallNode3821_covers smallNode3833_covers (by decide) (by decide) smallRow3823_covers

def smallNode3797 : Graph := .node 3797 smallRow3797 smallNode3779 smallNode3823
set_option maxRecDepth 4096 in
theorem smallNode3797_covers : Graph.Covers PrimePairVertices.smallTree smallNode3797 :=
  Graph.node_covers _ 3797 smallRow3797 smallNode3779 smallNode3823
    smallNode3779_covers smallNode3823_covers (by decide) (by decide) smallRow3797_covers

def smallNode3761 : Graph := .node 3761 smallRow3761 smallNode3709 smallNode3797
set_option maxRecDepth 4096 in
theorem smallNode3761_covers : Graph.Covers PrimePairVertices.smallTree smallNode3761 :=
  Graph.node_covers _ 3761 smallRow3761 smallNode3709 smallNode3797
    smallNode3709_covers smallNode3797_covers (by decide) (by decide) smallRow3761_covers

def smallNode3851 : Graph := .node 3851 smallRow3851 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3851_covers : Graph.Covers PrimePairVertices.smallTree smallNode3851 :=
  Graph.node_covers _ 3851 smallRow3851 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3851_covers

def smallNode3853 : Graph := .node 3853 smallRow3853 smallNode3851 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3853_covers : Graph.Covers PrimePairVertices.smallTree smallNode3853 :=
  Graph.node_covers _ 3853 smallRow3853 smallNode3851 smallEmpty
    smallNode3851_covers smallEmpty_covers (by decide) (by decide) smallRow3853_covers

def smallNode3877 : Graph := .node 3877 smallRow3877 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3877_covers : Graph.Covers PrimePairVertices.smallTree smallNode3877 :=
  Graph.node_covers _ 3877 smallRow3877 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3877_covers

def smallNode3863 : Graph := .node 3863 smallRow3863 smallNode3853 smallNode3877
set_option maxRecDepth 4096 in
theorem smallNode3863_covers : Graph.Covers PrimePairVertices.smallTree smallNode3863 :=
  Graph.node_covers _ 3863 smallRow3863 smallNode3853 smallNode3877
    smallNode3853_covers smallNode3877_covers (by decide) (by decide) smallRow3863_covers

def smallNode3889 : Graph := .node 3889 smallRow3889 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3889_covers : Graph.Covers PrimePairVertices.smallTree smallNode3889 :=
  Graph.node_covers _ 3889 smallRow3889 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3889_covers

def smallNode3907 : Graph := .node 3907 smallRow3907 smallNode3889 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3907_covers : Graph.Covers PrimePairVertices.smallTree smallNode3907 :=
  Graph.node_covers _ 3907 smallRow3907 smallNode3889 smallEmpty
    smallNode3889_covers smallEmpty_covers (by decide) (by decide) smallRow3907_covers

def smallNode3917 : Graph := .node 3917 smallRow3917 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3917_covers : Graph.Covers PrimePairVertices.smallTree smallNode3917 :=
  Graph.node_covers _ 3917 smallRow3917 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3917_covers

def smallNode3911 : Graph := .node 3911 smallRow3911 smallNode3907 smallNode3917
set_option maxRecDepth 4096 in
theorem smallNode3911_covers : Graph.Covers PrimePairVertices.smallTree smallNode3911 :=
  Graph.node_covers _ 3911 smallRow3911 smallNode3907 smallNode3917
    smallNode3907_covers smallNode3917_covers (by decide) (by decide) smallRow3911_covers

def smallNode3881 : Graph := .node 3881 smallRow3881 smallNode3863 smallNode3911
set_option maxRecDepth 4096 in
theorem smallNode3881_covers : Graph.Covers PrimePairVertices.smallTree smallNode3881 :=
  Graph.node_covers _ 3881 smallRow3881 smallNode3863 smallNode3911
    smallNode3863_covers smallNode3911_covers (by decide) (by decide) smallRow3881_covers

def smallNode3923 : Graph := .node 3923 smallRow3923 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3923_covers : Graph.Covers PrimePairVertices.smallTree smallNode3923 :=
  Graph.node_covers _ 3923 smallRow3923 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3923_covers

def smallNode3929 : Graph := .node 3929 smallRow3929 smallNode3923 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3929_covers : Graph.Covers PrimePairVertices.smallTree smallNode3929 :=
  Graph.node_covers _ 3929 smallRow3929 smallNode3923 smallEmpty
    smallNode3923_covers smallEmpty_covers (by decide) (by decide) smallRow3929_covers

def smallNode3943 : Graph := .node 3943 smallRow3943 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3943_covers : Graph.Covers PrimePairVertices.smallTree smallNode3943 :=
  Graph.node_covers _ 3943 smallRow3943 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3943_covers

def smallNode3931 : Graph := .node 3931 smallRow3931 smallNode3929 smallNode3943
set_option maxRecDepth 4096 in
theorem smallNode3931_covers : Graph.Covers PrimePairVertices.smallTree smallNode3931 :=
  Graph.node_covers _ 3931 smallRow3931 smallNode3929 smallNode3943
    smallNode3929_covers smallNode3943_covers (by decide) (by decide) smallRow3931_covers

def smallNode3967 : Graph := .node 3967 smallRow3967 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3967_covers : Graph.Covers PrimePairVertices.smallTree smallNode3967 :=
  Graph.node_covers _ 3967 smallRow3967 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow3967_covers

def smallNode3989 : Graph := .node 3989 smallRow3989 smallNode3967 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3989_covers : Graph.Covers PrimePairVertices.smallTree smallNode3989 :=
  Graph.node_covers _ 3989 smallRow3989 smallNode3967 smallEmpty
    smallNode3967_covers smallEmpty_covers (by decide) (by decide) smallRow3989_covers

def smallNode4003 : Graph := .node 4003 smallRow4003 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4003_covers : Graph.Covers PrimePairVertices.smallTree smallNode4003 :=
  Graph.node_covers _ 4003 smallRow4003 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4003_covers

def smallNode4001 : Graph := .node 4001 smallRow4001 smallNode3989 smallNode4003
set_option maxRecDepth 4096 in
theorem smallNode4001_covers : Graph.Covers PrimePairVertices.smallTree smallNode4001 :=
  Graph.node_covers _ 4001 smallRow4001 smallNode3989 smallNode4003
    smallNode3989_covers smallNode4003_covers (by decide) (by decide) smallRow4001_covers

def smallNode3947 : Graph := .node 3947 smallRow3947 smallNode3931 smallNode4001
set_option maxRecDepth 4096 in
theorem smallNode3947_covers : Graph.Covers PrimePairVertices.smallTree smallNode3947 :=
  Graph.node_covers _ 3947 smallRow3947 smallNode3931 smallNode4001
    smallNode3931_covers smallNode4001_covers (by decide) (by decide) smallRow3947_covers

def smallNode3919 : Graph := .node 3919 smallRow3919 smallNode3881 smallNode3947
set_option maxRecDepth 4096 in
theorem smallNode3919_covers : Graph.Covers PrimePairVertices.smallTree smallNode3919 :=
  Graph.node_covers _ 3919 smallRow3919 smallNode3881 smallNode3947
    smallNode3881_covers smallNode3947_covers (by decide) (by decide) smallRow3919_covers

def smallNode3847 : Graph := .node 3847 smallRow3847 smallNode3761 smallNode3919
set_option maxRecDepth 4096 in
theorem smallNode3847_covers : Graph.Covers PrimePairVertices.smallTree smallNode3847 :=
  Graph.node_covers _ 3847 smallRow3847 smallNode3761 smallNode3919
    smallNode3761_covers smallNode3919_covers (by decide) (by decide) smallRow3847_covers

def smallNode3671 : Graph := .node 3671 smallRow3671 smallNode3517 smallNode3847
set_option maxRecDepth 4096 in
theorem smallNode3671_covers : Graph.Covers PrimePairVertices.smallTree smallNode3671 :=
  Graph.node_covers _ 3671 smallRow3671 smallNode3517 smallNode3847
    smallNode3517_covers smallNode3847_covers (by decide) (by decide) smallRow3671_covers

def smallNode3331 : Graph := .node 3331 smallRow3331 smallNode2969 smallNode3671
set_option maxRecDepth 4096 in
theorem smallNode3331_covers : Graph.Covers PrimePairVertices.smallTree smallNode3331 :=
  Graph.node_covers _ 3331 smallRow3331 smallNode2969 smallNode3671
    smallNode2969_covers smallNode3671_covers (by decide) (by decide) smallRow3331_covers

def smallNode4013 : Graph := .node 4013 smallRow4013 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4013_covers : Graph.Covers PrimePairVertices.smallTree smallNode4013 :=
  Graph.node_covers _ 4013 smallRow4013 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4013_covers

def smallNode4019 : Graph := .node 4019 smallRow4019 smallNode4013 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4019_covers : Graph.Covers PrimePairVertices.smallTree smallNode4019 :=
  Graph.node_covers _ 4019 smallRow4019 smallNode4013 smallEmpty
    smallNode4013_covers smallEmpty_covers (by decide) (by decide) smallRow4019_covers

def smallNode4027 : Graph := .node 4027 smallRow4027 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4027_covers : Graph.Covers PrimePairVertices.smallTree smallNode4027 :=
  Graph.node_covers _ 4027 smallRow4027 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4027_covers

def smallNode4049 : Graph := .node 4049 smallRow4049 smallNode4027 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4049_covers : Graph.Covers PrimePairVertices.smallTree smallNode4049 :=
  Graph.node_covers _ 4049 smallRow4049 smallNode4027 smallEmpty
    smallNode4027_covers smallEmpty_covers (by decide) (by decide) smallRow4049_covers

def smallNode4021 : Graph := .node 4021 smallRow4021 smallNode4019 smallNode4049
set_option maxRecDepth 4096 in
theorem smallNode4021_covers : Graph.Covers PrimePairVertices.smallTree smallNode4021 :=
  Graph.node_covers _ 4021 smallRow4021 smallNode4019 smallNode4049
    smallNode4019_covers smallNode4049_covers (by decide) (by decide) smallRow4021_covers

def smallNode4057 : Graph := .node 4057 smallRow4057 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4057_covers : Graph.Covers PrimePairVertices.smallTree smallNode4057 :=
  Graph.node_covers _ 4057 smallRow4057 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4057_covers

def smallNode4073 : Graph := .node 4073 smallRow4073 smallNode4057 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4073_covers : Graph.Covers PrimePairVertices.smallTree smallNode4073 :=
  Graph.node_covers _ 4073 smallRow4073 smallNode4057 smallEmpty
    smallNode4057_covers smallEmpty_covers (by decide) (by decide) smallRow4073_covers

def smallNode4091 : Graph := .node 4091 smallRow4091 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4091_covers : Graph.Covers PrimePairVertices.smallTree smallNode4091 :=
  Graph.node_covers _ 4091 smallRow4091 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4091_covers

def smallNode4079 : Graph := .node 4079 smallRow4079 smallNode4073 smallNode4091
set_option maxRecDepth 4096 in
theorem smallNode4079_covers : Graph.Covers PrimePairVertices.smallTree smallNode4079 :=
  Graph.node_covers _ 4079 smallRow4079 smallNode4073 smallNode4091
    smallNode4073_covers smallNode4091_covers (by decide) (by decide) smallRow4079_covers

def smallNode4051 : Graph := .node 4051 smallRow4051 smallNode4021 smallNode4079
set_option maxRecDepth 4096 in
theorem smallNode4051_covers : Graph.Covers PrimePairVertices.smallTree smallNode4051 :=
  Graph.node_covers _ 4051 smallRow4051 smallNode4021 smallNode4079
    smallNode4021_covers smallNode4079_covers (by decide) (by decide) smallRow4051_covers

def smallNode4099 : Graph := .node 4099 smallRow4099 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4099_covers : Graph.Covers PrimePairVertices.smallTree smallNode4099 :=
  Graph.node_covers _ 4099 smallRow4099 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4099_covers

def smallNode4111 : Graph := .node 4111 smallRow4111 smallNode4099 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4111_covers : Graph.Covers PrimePairVertices.smallTree smallNode4111 :=
  Graph.node_covers _ 4111 smallRow4111 smallNode4099 smallEmpty
    smallNode4099_covers smallEmpty_covers (by decide) (by decide) smallRow4111_covers

def smallNode4129 : Graph := .node 4129 smallRow4129 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4129_covers : Graph.Covers PrimePairVertices.smallTree smallNode4129 :=
  Graph.node_covers _ 4129 smallRow4129 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4129_covers

def smallNode4127 : Graph := .node 4127 smallRow4127 smallNode4111 smallNode4129
set_option maxRecDepth 4096 in
theorem smallNode4127_covers : Graph.Covers PrimePairVertices.smallTree smallNode4127 :=
  Graph.node_covers _ 4127 smallRow4127 smallNode4111 smallNode4129
    smallNode4111_covers smallNode4129_covers (by decide) (by decide) smallRow4127_covers

def smallNode4139 : Graph := .node 4139 smallRow4139 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4139_covers : Graph.Covers PrimePairVertices.smallTree smallNode4139 :=
  Graph.node_covers _ 4139 smallRow4139 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4139_covers

def smallNode4153 : Graph := .node 4153 smallRow4153 smallNode4139 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4153_covers : Graph.Covers PrimePairVertices.smallTree smallNode4153 :=
  Graph.node_covers _ 4153 smallRow4153 smallNode4139 smallEmpty
    smallNode4139_covers smallEmpty_covers (by decide) (by decide) smallRow4153_covers

def smallNode4159 : Graph := .node 4159 smallRow4159 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4159_covers : Graph.Covers PrimePairVertices.smallTree smallNode4159 :=
  Graph.node_covers _ 4159 smallRow4159 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4159_covers

def smallNode4157 : Graph := .node 4157 smallRow4157 smallNode4153 smallNode4159
set_option maxRecDepth 4096 in
theorem smallNode4157_covers : Graph.Covers PrimePairVertices.smallTree smallNode4157 :=
  Graph.node_covers _ 4157 smallRow4157 smallNode4153 smallNode4159
    smallNode4153_covers smallNode4159_covers (by decide) (by decide) smallRow4157_covers

def smallNode4133 : Graph := .node 4133 smallRow4133 smallNode4127 smallNode4157
set_option maxRecDepth 4096 in
theorem smallNode4133_covers : Graph.Covers PrimePairVertices.smallTree smallNode4133 :=
  Graph.node_covers _ 4133 smallRow4133 smallNode4127 smallNode4157
    smallNode4127_covers smallNode4157_covers (by decide) (by decide) smallRow4133_covers

def smallNode4093 : Graph := .node 4093 smallRow4093 smallNode4051 smallNode4133
set_option maxRecDepth 4096 in
theorem smallNode4093_covers : Graph.Covers PrimePairVertices.smallTree smallNode4093 :=
  Graph.node_covers _ 4093 smallRow4093 smallNode4051 smallNode4133
    smallNode4051_covers smallNode4133_covers (by decide) (by decide) smallRow4093_covers

def smallNode4201 : Graph := .node 4201 smallRow4201 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4201_covers : Graph.Covers PrimePairVertices.smallTree smallNode4201 :=
  Graph.node_covers _ 4201 smallRow4201 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4201_covers

def smallNode4211 : Graph := .node 4211 smallRow4211 smallNode4201 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4211_covers : Graph.Covers PrimePairVertices.smallTree smallNode4211 :=
  Graph.node_covers _ 4211 smallRow4211 smallNode4201 smallEmpty
    smallNode4201_covers smallEmpty_covers (by decide) (by decide) smallRow4211_covers

def smallNode4219 : Graph := .node 4219 smallRow4219 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4219_covers : Graph.Covers PrimePairVertices.smallTree smallNode4219 :=
  Graph.node_covers _ 4219 smallRow4219 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4219_covers

def smallNode4229 : Graph := .node 4229 smallRow4229 smallNode4219 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4229_covers : Graph.Covers PrimePairVertices.smallTree smallNode4229 :=
  Graph.node_covers _ 4229 smallRow4229 smallNode4219 smallEmpty
    smallNode4219_covers smallEmpty_covers (by decide) (by decide) smallRow4229_covers

def smallNode4217 : Graph := .node 4217 smallRow4217 smallNode4211 smallNode4229
set_option maxRecDepth 4096 in
theorem smallNode4217_covers : Graph.Covers PrimePairVertices.smallTree smallNode4217 :=
  Graph.node_covers _ 4217 smallRow4217 smallNode4211 smallNode4229
    smallNode4211_covers smallNode4229_covers (by decide) (by decide) smallRow4217_covers

def smallNode4241 : Graph := .node 4241 smallRow4241 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4241_covers : Graph.Covers PrimePairVertices.smallTree smallNode4241 :=
  Graph.node_covers _ 4241 smallRow4241 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4241_covers

def smallNode4243 : Graph := .node 4243 smallRow4243 smallNode4241 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4243_covers : Graph.Covers PrimePairVertices.smallTree smallNode4243 :=
  Graph.node_covers _ 4243 smallRow4243 smallNode4241 smallEmpty
    smallNode4241_covers smallEmpty_covers (by decide) (by decide) smallRow4243_covers

def smallNode4259 : Graph := .node 4259 smallRow4259 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4259_covers : Graph.Covers PrimePairVertices.smallTree smallNode4259 :=
  Graph.node_covers _ 4259 smallRow4259 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4259_covers

def smallNode4253 : Graph := .node 4253 smallRow4253 smallNode4243 smallNode4259
set_option maxRecDepth 4096 in
theorem smallNode4253_covers : Graph.Covers PrimePairVertices.smallTree smallNode4253 :=
  Graph.node_covers _ 4253 smallRow4253 smallNode4243 smallNode4259
    smallNode4243_covers smallNode4259_covers (by decide) (by decide) smallRow4253_covers

def smallNode4231 : Graph := .node 4231 smallRow4231 smallNode4217 smallNode4253
set_option maxRecDepth 4096 in
theorem smallNode4231_covers : Graph.Covers PrimePairVertices.smallTree smallNode4231 :=
  Graph.node_covers _ 4231 smallRow4231 smallNode4217 smallNode4253
    smallNode4217_covers smallNode4253_covers (by decide) (by decide) smallRow4231_covers

def smallNode4271 : Graph := .node 4271 smallRow4271 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4271_covers : Graph.Covers PrimePairVertices.smallTree smallNode4271 :=
  Graph.node_covers _ 4271 smallRow4271 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4271_covers

def smallNode4273 : Graph := .node 4273 smallRow4273 smallNode4271 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4273_covers : Graph.Covers PrimePairVertices.smallTree smallNode4273 :=
  Graph.node_covers _ 4273 smallRow4273 smallNode4271 smallEmpty
    smallNode4271_covers smallEmpty_covers (by decide) (by decide) smallRow4273_covers

def smallNode4289 : Graph := .node 4289 smallRow4289 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4289_covers : Graph.Covers PrimePairVertices.smallTree smallNode4289 :=
  Graph.node_covers _ 4289 smallRow4289 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4289_covers

def smallNode4283 : Graph := .node 4283 smallRow4283 smallNode4273 smallNode4289
set_option maxRecDepth 4096 in
theorem smallNode4283_covers : Graph.Covers PrimePairVertices.smallTree smallNode4283 :=
  Graph.node_covers _ 4283 smallRow4283 smallNode4273 smallNode4289
    smallNode4273_covers smallNode4289_covers (by decide) (by decide) smallRow4283_covers

def smallNode4327 : Graph := .node 4327 smallRow4327 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4327_covers : Graph.Covers PrimePairVertices.smallTree smallNode4327 :=
  Graph.node_covers _ 4327 smallRow4327 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4327_covers

def smallNode4337 : Graph := .node 4337 smallRow4337 smallNode4327 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4337_covers : Graph.Covers PrimePairVertices.smallTree smallNode4337 :=
  Graph.node_covers _ 4337 smallRow4337 smallNode4327 smallEmpty
    smallNode4327_covers smallEmpty_covers (by decide) (by decide) smallRow4337_covers

def smallNode4349 : Graph := .node 4349 smallRow4349 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4349_covers : Graph.Covers PrimePairVertices.smallTree smallNode4349 :=
  Graph.node_covers _ 4349 smallRow4349 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4349_covers

def smallNode4339 : Graph := .node 4339 smallRow4339 smallNode4337 smallNode4349
set_option maxRecDepth 4096 in
theorem smallNode4339_covers : Graph.Covers PrimePairVertices.smallTree smallNode4339 :=
  Graph.node_covers _ 4339 smallRow4339 smallNode4337 smallNode4349
    smallNode4337_covers smallNode4349_covers (by decide) (by decide) smallRow4339_covers

def smallNode4297 : Graph := .node 4297 smallRow4297 smallNode4283 smallNode4339
set_option maxRecDepth 4096 in
theorem smallNode4297_covers : Graph.Covers PrimePairVertices.smallTree smallNode4297 :=
  Graph.node_covers _ 4297 smallRow4297 smallNode4283 smallNode4339
    smallNode4283_covers smallNode4339_covers (by decide) (by decide) smallRow4297_covers

def smallNode4261 : Graph := .node 4261 smallRow4261 smallNode4231 smallNode4297
set_option maxRecDepth 4096 in
theorem smallNode4261_covers : Graph.Covers PrimePairVertices.smallTree smallNode4261 :=
  Graph.node_covers _ 4261 smallRow4261 smallNode4231 smallNode4297
    smallNode4231_covers smallNode4297_covers (by decide) (by decide) smallRow4261_covers

def smallNode4177 : Graph := .node 4177 smallRow4177 smallNode4093 smallNode4261
set_option maxRecDepth 4096 in
theorem smallNode4177_covers : Graph.Covers PrimePairVertices.smallTree smallNode4177 :=
  Graph.node_covers _ 4177 smallRow4177 smallNode4093 smallNode4261
    smallNode4093_covers smallNode4261_covers (by decide) (by decide) smallRow4177_covers

def smallNode4363 : Graph := .node 4363 smallRow4363 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4363_covers : Graph.Covers PrimePairVertices.smallTree smallNode4363 :=
  Graph.node_covers _ 4363 smallRow4363 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4363_covers

def smallNode4373 : Graph := .node 4373 smallRow4373 smallNode4363 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4373_covers : Graph.Covers PrimePairVertices.smallTree smallNode4373 :=
  Graph.node_covers _ 4373 smallRow4373 smallNode4363 smallEmpty
    smallNode4363_covers smallEmpty_covers (by decide) (by decide) smallRow4373_covers

def smallNode4397 : Graph := .node 4397 smallRow4397 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4397_covers : Graph.Covers PrimePairVertices.smallTree smallNode4397 :=
  Graph.node_covers _ 4397 smallRow4397 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4397_covers

def smallNode4409 : Graph := .node 4409 smallRow4409 smallNode4397 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4409_covers : Graph.Covers PrimePairVertices.smallTree smallNode4409 :=
  Graph.node_covers _ 4409 smallRow4409 smallNode4397 smallEmpty
    smallNode4397_covers smallEmpty_covers (by decide) (by decide) smallRow4409_covers

def smallNode4391 : Graph := .node 4391 smallRow4391 smallNode4373 smallNode4409
set_option maxRecDepth 4096 in
theorem smallNode4391_covers : Graph.Covers PrimePairVertices.smallTree smallNode4391 :=
  Graph.node_covers _ 4391 smallRow4391 smallNode4373 smallNode4409
    smallNode4373_covers smallNode4409_covers (by decide) (by decide) smallRow4391_covers

def smallNode4423 : Graph := .node 4423 smallRow4423 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4423_covers : Graph.Covers PrimePairVertices.smallTree smallNode4423 :=
  Graph.node_covers _ 4423 smallRow4423 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4423_covers

def smallNode4441 : Graph := .node 4441 smallRow4441 smallNode4423 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4441_covers : Graph.Covers PrimePairVertices.smallTree smallNode4441 :=
  Graph.node_covers _ 4441 smallRow4441 smallNode4423 smallEmpty
    smallNode4423_covers smallEmpty_covers (by decide) (by decide) smallRow4441_covers

def smallNode4451 : Graph := .node 4451 smallRow4451 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4451_covers : Graph.Covers PrimePairVertices.smallTree smallNode4451 :=
  Graph.node_covers _ 4451 smallRow4451 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4451_covers

def smallNode4447 : Graph := .node 4447 smallRow4447 smallNode4441 smallNode4451
set_option maxRecDepth 4096 in
theorem smallNode4447_covers : Graph.Covers PrimePairVertices.smallTree smallNode4447 :=
  Graph.node_covers _ 4447 smallRow4447 smallNode4441 smallNode4451
    smallNode4441_covers smallNode4451_covers (by decide) (by decide) smallRow4447_covers

def smallNode4421 : Graph := .node 4421 smallRow4421 smallNode4391 smallNode4447
set_option maxRecDepth 4096 in
theorem smallNode4421_covers : Graph.Covers PrimePairVertices.smallTree smallNode4421 :=
  Graph.node_covers _ 4421 smallRow4421 smallNode4391 smallNode4447
    smallNode4391_covers smallNode4447_covers (by decide) (by decide) smallRow4421_covers

def smallNode4463 : Graph := .node 4463 smallRow4463 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4463_covers : Graph.Covers PrimePairVertices.smallTree smallNode4463 :=
  Graph.node_covers _ 4463 smallRow4463 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4463_covers

def smallNode4481 : Graph := .node 4481 smallRow4481 smallNode4463 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4481_covers : Graph.Covers PrimePairVertices.smallTree smallNode4481 :=
  Graph.node_covers _ 4481 smallRow4481 smallNode4463 smallEmpty
    smallNode4463_covers smallEmpty_covers (by decide) (by decide) smallRow4481_covers

def smallNode4493 : Graph := .node 4493 smallRow4493 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4493_covers : Graph.Covers PrimePairVertices.smallTree smallNode4493 :=
  Graph.node_covers _ 4493 smallRow4493 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4493_covers

def smallNode4483 : Graph := .node 4483 smallRow4483 smallNode4481 smallNode4493
set_option maxRecDepth 4096 in
theorem smallNode4483_covers : Graph.Covers PrimePairVertices.smallTree smallNode4483 :=
  Graph.node_covers _ 4483 smallRow4483 smallNode4481 smallNode4493
    smallNode4481_covers smallNode4493_covers (by decide) (by decide) smallRow4483_covers

def smallNode4513 : Graph := .node 4513 smallRow4513 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4513_covers : Graph.Covers PrimePairVertices.smallTree smallNode4513 :=
  Graph.node_covers _ 4513 smallRow4513 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4513_covers

def smallNode4517 : Graph := .node 4517 smallRow4517 smallNode4513 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4517_covers : Graph.Covers PrimePairVertices.smallTree smallNode4517 :=
  Graph.node_covers _ 4517 smallRow4517 smallNode4513 smallEmpty
    smallNode4513_covers smallEmpty_covers (by decide) (by decide) smallRow4517_covers

def smallNode4523 : Graph := .node 4523 smallRow4523 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4523_covers : Graph.Covers PrimePairVertices.smallTree smallNode4523 :=
  Graph.node_covers _ 4523 smallRow4523 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4523_covers

def smallNode4519 : Graph := .node 4519 smallRow4519 smallNode4517 smallNode4523
set_option maxRecDepth 4096 in
theorem smallNode4519_covers : Graph.Covers PrimePairVertices.smallTree smallNode4519 :=
  Graph.node_covers _ 4519 smallRow4519 smallNode4517 smallNode4523
    smallNode4517_covers smallNode4523_covers (by decide) (by decide) smallRow4519_covers

def smallNode4507 : Graph := .node 4507 smallRow4507 smallNode4483 smallNode4519
set_option maxRecDepth 4096 in
theorem smallNode4507_covers : Graph.Covers PrimePairVertices.smallTree smallNode4507 :=
  Graph.node_covers _ 4507 smallRow4507 smallNode4483 smallNode4519
    smallNode4483_covers smallNode4519_covers (by decide) (by decide) smallRow4507_covers

def smallNode4457 : Graph := .node 4457 smallRow4457 smallNode4421 smallNode4507
set_option maxRecDepth 4096 in
theorem smallNode4457_covers : Graph.Covers PrimePairVertices.smallTree smallNode4457 :=
  Graph.node_covers _ 4457 smallRow4457 smallNode4421 smallNode4507
    smallNode4421_covers smallNode4507_covers (by decide) (by decide) smallRow4457_covers

def smallNode4549 : Graph := .node 4549 smallRow4549 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4549_covers : Graph.Covers PrimePairVertices.smallTree smallNode4549 :=
  Graph.node_covers _ 4549 smallRow4549 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4549_covers

def smallNode4561 : Graph := .node 4561 smallRow4561 smallNode4549 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4561_covers : Graph.Covers PrimePairVertices.smallTree smallNode4561 :=
  Graph.node_covers _ 4561 smallRow4561 smallNode4549 smallEmpty
    smallNode4549_covers smallEmpty_covers (by decide) (by decide) smallRow4561_covers

def smallNode4583 : Graph := .node 4583 smallRow4583 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4583_covers : Graph.Covers PrimePairVertices.smallTree smallNode4583 :=
  Graph.node_covers _ 4583 smallRow4583 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4583_covers

def smallNode4567 : Graph := .node 4567 smallRow4567 smallNode4561 smallNode4583
set_option maxRecDepth 4096 in
theorem smallNode4567_covers : Graph.Covers PrimePairVertices.smallTree smallNode4567 :=
  Graph.node_covers _ 4567 smallRow4567 smallNode4561 smallNode4583
    smallNode4561_covers smallNode4583_covers (by decide) (by decide) smallRow4567_covers

def smallNode4597 : Graph := .node 4597 smallRow4597 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4597_covers : Graph.Covers PrimePairVertices.smallTree smallNode4597 :=
  Graph.node_covers _ 4597 smallRow4597 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4597_covers

def smallNode4603 : Graph := .node 4603 smallRow4603 smallNode4597 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4603_covers : Graph.Covers PrimePairVertices.smallTree smallNode4603 :=
  Graph.node_covers _ 4603 smallRow4603 smallNode4597 smallEmpty
    smallNode4597_covers smallEmpty_covers (by decide) (by decide) smallRow4603_covers

def smallNode4637 : Graph := .node 4637 smallRow4637 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4637_covers : Graph.Covers PrimePairVertices.smallTree smallNode4637 :=
  Graph.node_covers _ 4637 smallRow4637 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4637_covers

def smallNode4621 : Graph := .node 4621 smallRow4621 smallNode4603 smallNode4637
set_option maxRecDepth 4096 in
theorem smallNode4621_covers : Graph.Covers PrimePairVertices.smallTree smallNode4621 :=
  Graph.node_covers _ 4621 smallRow4621 smallNode4603 smallNode4637
    smallNode4603_covers smallNode4637_covers (by decide) (by decide) smallRow4621_covers

def smallNode4591 : Graph := .node 4591 smallRow4591 smallNode4567 smallNode4621
set_option maxRecDepth 4096 in
theorem smallNode4591_covers : Graph.Covers PrimePairVertices.smallTree smallNode4591 :=
  Graph.node_covers _ 4591 smallRow4591 smallNode4567 smallNode4621
    smallNode4567_covers smallNode4621_covers (by decide) (by decide) smallRow4591_covers

def smallNode4643 : Graph := .node 4643 smallRow4643 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4643_covers : Graph.Covers PrimePairVertices.smallTree smallNode4643 :=
  Graph.node_covers _ 4643 smallRow4643 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4643_covers

def smallNode4649 : Graph := .node 4649 smallRow4649 smallNode4643 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4649_covers : Graph.Covers PrimePairVertices.smallTree smallNode4649 :=
  Graph.node_covers _ 4649 smallRow4649 smallNode4643 smallEmpty
    smallNode4643_covers smallEmpty_covers (by decide) (by decide) smallRow4649_covers

def smallNode4657 : Graph := .node 4657 smallRow4657 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4657_covers : Graph.Covers PrimePairVertices.smallTree smallNode4657 :=
  Graph.node_covers _ 4657 smallRow4657 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4657_covers

def smallNode4651 : Graph := .node 4651 smallRow4651 smallNode4649 smallNode4657
set_option maxRecDepth 4096 in
theorem smallNode4651_covers : Graph.Covers PrimePairVertices.smallTree smallNode4651 :=
  Graph.node_covers _ 4651 smallRow4651 smallNode4649 smallNode4657
    smallNode4649_covers smallNode4657_covers (by decide) (by decide) smallRow4651_covers

def smallNode4673 : Graph := .node 4673 smallRow4673 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4673_covers : Graph.Covers PrimePairVertices.smallTree smallNode4673 :=
  Graph.node_covers _ 4673 smallRow4673 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4673_covers

def smallNode4679 : Graph := .node 4679 smallRow4679 smallNode4673 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4679_covers : Graph.Covers PrimePairVertices.smallTree smallNode4679 :=
  Graph.node_covers _ 4679 smallRow4679 smallNode4673 smallEmpty
    smallNode4673_covers smallEmpty_covers (by decide) (by decide) smallRow4679_covers

def smallNode4703 : Graph := .node 4703 smallRow4703 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4703_covers : Graph.Covers PrimePairVertices.smallTree smallNode4703 :=
  Graph.node_covers _ 4703 smallRow4703 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4703_covers

def smallNode4691 : Graph := .node 4691 smallRow4691 smallNode4679 smallNode4703
set_option maxRecDepth 4096 in
theorem smallNode4691_covers : Graph.Covers PrimePairVertices.smallTree smallNode4691 :=
  Graph.node_covers _ 4691 smallRow4691 smallNode4679 smallNode4703
    smallNode4679_covers smallNode4703_covers (by decide) (by decide) smallRow4691_covers

def smallNode4663 : Graph := .node 4663 smallRow4663 smallNode4651 smallNode4691
set_option maxRecDepth 4096 in
theorem smallNode4663_covers : Graph.Covers PrimePairVertices.smallTree smallNode4663 :=
  Graph.node_covers _ 4663 smallRow4663 smallNode4651 smallNode4691
    smallNode4651_covers smallNode4691_covers (by decide) (by decide) smallRow4663_covers

def smallNode4639 : Graph := .node 4639 smallRow4639 smallNode4591 smallNode4663
set_option maxRecDepth 4096 in
theorem smallNode4639_covers : Graph.Covers PrimePairVertices.smallTree smallNode4639 :=
  Graph.node_covers _ 4639 smallRow4639 smallNode4591 smallNode4663
    smallNode4591_covers smallNode4663_covers (by decide) (by decide) smallRow4639_covers

def smallNode4547 : Graph := .node 4547 smallRow4547 smallNode4457 smallNode4639
set_option maxRecDepth 4096 in
theorem smallNode4547_covers : Graph.Covers PrimePairVertices.smallTree smallNode4547 :=
  Graph.node_covers _ 4547 smallRow4547 smallNode4457 smallNode4639
    smallNode4457_covers smallNode4639_covers (by decide) (by decide) smallRow4547_covers

def smallNode4357 : Graph := .node 4357 smallRow4357 smallNode4177 smallNode4547
set_option maxRecDepth 4096 in
theorem smallNode4357_covers : Graph.Covers PrimePairVertices.smallTree smallNode4357 :=
  Graph.node_covers _ 4357 smallRow4357 smallNode4177 smallNode4547
    smallNode4177_covers smallNode4547_covers (by decide) (by decide) smallRow4357_covers

def smallNode4723 : Graph := .node 4723 smallRow4723 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4723_covers : Graph.Covers PrimePairVertices.smallTree smallNode4723 :=
  Graph.node_covers _ 4723 smallRow4723 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4723_covers

def smallNode4729 : Graph := .node 4729 smallRow4729 smallNode4723 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4729_covers : Graph.Covers PrimePairVertices.smallTree smallNode4729 :=
  Graph.node_covers _ 4729 smallRow4729 smallNode4723 smallEmpty
    smallNode4723_covers smallEmpty_covers (by decide) (by decide) smallRow4729_covers

def smallNode4751 : Graph := .node 4751 smallRow4751 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4751_covers : Graph.Covers PrimePairVertices.smallTree smallNode4751 :=
  Graph.node_covers _ 4751 smallRow4751 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4751_covers

def smallNode4759 : Graph := .node 4759 smallRow4759 smallNode4751 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4759_covers : Graph.Covers PrimePairVertices.smallTree smallNode4759 :=
  Graph.node_covers _ 4759 smallRow4759 smallNode4751 smallEmpty
    smallNode4751_covers smallEmpty_covers (by decide) (by decide) smallRow4759_covers

def smallNode4733 : Graph := .node 4733 smallRow4733 smallNode4729 smallNode4759
set_option maxRecDepth 4096 in
theorem smallNode4733_covers : Graph.Covers PrimePairVertices.smallTree smallNode4733 :=
  Graph.node_covers _ 4733 smallRow4733 smallNode4729 smallNode4759
    smallNode4729_covers smallNode4759_covers (by decide) (by decide) smallRow4733_covers

def smallNode4787 : Graph := .node 4787 smallRow4787 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4787_covers : Graph.Covers PrimePairVertices.smallTree smallNode4787 :=
  Graph.node_covers _ 4787 smallRow4787 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4787_covers

def smallNode4789 : Graph := .node 4789 smallRow4789 smallNode4787 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4789_covers : Graph.Covers PrimePairVertices.smallTree smallNode4789 :=
  Graph.node_covers _ 4789 smallRow4789 smallNode4787 smallEmpty
    smallNode4787_covers smallEmpty_covers (by decide) (by decide) smallRow4789_covers

def smallNode4799 : Graph := .node 4799 smallRow4799 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4799_covers : Graph.Covers PrimePairVertices.smallTree smallNode4799 :=
  Graph.node_covers _ 4799 smallRow4799 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4799_covers

def smallNode4793 : Graph := .node 4793 smallRow4793 smallNode4789 smallNode4799
set_option maxRecDepth 4096 in
theorem smallNode4793_covers : Graph.Covers PrimePairVertices.smallTree smallNode4793 :=
  Graph.node_covers _ 4793 smallRow4793 smallNode4789 smallNode4799
    smallNode4789_covers smallNode4799_covers (by decide) (by decide) smallRow4793_covers

def smallNode4783 : Graph := .node 4783 smallRow4783 smallNode4733 smallNode4793
set_option maxRecDepth 4096 in
theorem smallNode4783_covers : Graph.Covers PrimePairVertices.smallTree smallNode4783 :=
  Graph.node_covers _ 4783 smallRow4783 smallNode4733 smallNode4793
    smallNode4733_covers smallNode4793_covers (by decide) (by decide) smallRow4783_covers

def smallNode4813 : Graph := .node 4813 smallRow4813 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4813_covers : Graph.Covers PrimePairVertices.smallTree smallNode4813 :=
  Graph.node_covers _ 4813 smallRow4813 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4813_covers

def smallNode4817 : Graph := .node 4817 smallRow4817 smallNode4813 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4817_covers : Graph.Covers PrimePairVertices.smallTree smallNode4817 :=
  Graph.node_covers _ 4817 smallRow4817 smallNode4813 smallEmpty
    smallNode4813_covers smallEmpty_covers (by decide) (by decide) smallRow4817_covers

def smallNode4861 : Graph := .node 4861 smallRow4861 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4861_covers : Graph.Covers PrimePairVertices.smallTree smallNode4861 :=
  Graph.node_covers _ 4861 smallRow4861 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4861_covers

def smallNode4831 : Graph := .node 4831 smallRow4831 smallNode4817 smallNode4861
set_option maxRecDepth 4096 in
theorem smallNode4831_covers : Graph.Covers PrimePairVertices.smallTree smallNode4831 :=
  Graph.node_covers _ 4831 smallRow4831 smallNode4817 smallNode4861
    smallNode4817_covers smallNode4861_covers (by decide) (by decide) smallRow4831_covers

def smallNode4877 : Graph := .node 4877 smallRow4877 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4877_covers : Graph.Covers PrimePairVertices.smallTree smallNode4877 :=
  Graph.node_covers _ 4877 smallRow4877 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4877_covers

def smallNode4889 : Graph := .node 4889 smallRow4889 smallNode4877 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4889_covers : Graph.Covers PrimePairVertices.smallTree smallNode4889 :=
  Graph.node_covers _ 4889 smallRow4889 smallNode4877 smallEmpty
    smallNode4877_covers smallEmpty_covers (by decide) (by decide) smallRow4889_covers

def smallNode4909 : Graph := .node 4909 smallRow4909 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4909_covers : Graph.Covers PrimePairVertices.smallTree smallNode4909 :=
  Graph.node_covers _ 4909 smallRow4909 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4909_covers

def smallNode4903 : Graph := .node 4903 smallRow4903 smallNode4889 smallNode4909
set_option maxRecDepth 4096 in
theorem smallNode4903_covers : Graph.Covers PrimePairVertices.smallTree smallNode4903 :=
  Graph.node_covers _ 4903 smallRow4903 smallNode4889 smallNode4909
    smallNode4889_covers smallNode4909_covers (by decide) (by decide) smallRow4903_covers

def smallNode4871 : Graph := .node 4871 smallRow4871 smallNode4831 smallNode4903
set_option maxRecDepth 4096 in
theorem smallNode4871_covers : Graph.Covers PrimePairVertices.smallTree smallNode4871 :=
  Graph.node_covers _ 4871 smallRow4871 smallNode4831 smallNode4903
    smallNode4831_covers smallNode4903_covers (by decide) (by decide) smallRow4871_covers

def smallNode4801 : Graph := .node 4801 smallRow4801 smallNode4783 smallNode4871
set_option maxRecDepth 4096 in
theorem smallNode4801_covers : Graph.Covers PrimePairVertices.smallTree smallNode4801 :=
  Graph.node_covers _ 4801 smallRow4801 smallNode4783 smallNode4871
    smallNode4783_covers smallNode4871_covers (by decide) (by decide) smallRow4801_covers

def smallNode4931 : Graph := .node 4931 smallRow4931 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4931_covers : Graph.Covers PrimePairVertices.smallTree smallNode4931 :=
  Graph.node_covers _ 4931 smallRow4931 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4931_covers

def smallNode4933 : Graph := .node 4933 smallRow4933 smallNode4931 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4933_covers : Graph.Covers PrimePairVertices.smallTree smallNode4933 :=
  Graph.node_covers _ 4933 smallRow4933 smallNode4931 smallEmpty
    smallNode4931_covers smallEmpty_covers (by decide) (by decide) smallRow4933_covers

def smallNode4943 : Graph := .node 4943 smallRow4943 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4943_covers : Graph.Covers PrimePairVertices.smallTree smallNode4943 :=
  Graph.node_covers _ 4943 smallRow4943 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4943_covers

def smallNode4951 : Graph := .node 4951 smallRow4951 smallNode4943 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4951_covers : Graph.Covers PrimePairVertices.smallTree smallNode4951 :=
  Graph.node_covers _ 4951 smallRow4951 smallNode4943 smallEmpty
    smallNode4943_covers smallEmpty_covers (by decide) (by decide) smallRow4951_covers

def smallNode4937 : Graph := .node 4937 smallRow4937 smallNode4933 smallNode4951
set_option maxRecDepth 4096 in
theorem smallNode4937_covers : Graph.Covers PrimePairVertices.smallTree smallNode4937 :=
  Graph.node_covers _ 4937 smallRow4937 smallNode4933 smallNode4951
    smallNode4933_covers smallNode4951_covers (by decide) (by decide) smallRow4937_covers

def smallNode4967 : Graph := .node 4967 smallRow4967 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4967_covers : Graph.Covers PrimePairVertices.smallTree smallNode4967 :=
  Graph.node_covers _ 4967 smallRow4967 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4967_covers

def smallNode4969 : Graph := .node 4969 smallRow4969 smallNode4967 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4969_covers : Graph.Covers PrimePairVertices.smallTree smallNode4969 :=
  Graph.node_covers _ 4969 smallRow4969 smallNode4967 smallEmpty
    smallNode4967_covers smallEmpty_covers (by decide) (by decide) smallRow4969_covers

def smallNode4987 : Graph := .node 4987 smallRow4987 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4987_covers : Graph.Covers PrimePairVertices.smallTree smallNode4987 :=
  Graph.node_covers _ 4987 smallRow4987 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4987_covers

def smallNode4973 : Graph := .node 4973 smallRow4973 smallNode4969 smallNode4987
set_option maxRecDepth 4096 in
theorem smallNode4973_covers : Graph.Covers PrimePairVertices.smallTree smallNode4973 :=
  Graph.node_covers _ 4973 smallRow4973 smallNode4969 smallNode4987
    smallNode4969_covers smallNode4987_covers (by decide) (by decide) smallRow4973_covers

def smallNode4957 : Graph := .node 4957 smallRow4957 smallNode4937 smallNode4973
set_option maxRecDepth 4096 in
theorem smallNode4957_covers : Graph.Covers PrimePairVertices.smallTree smallNode4957 :=
  Graph.node_covers _ 4957 smallRow4957 smallNode4937 smallNode4973
    smallNode4937_covers smallNode4973_covers (by decide) (by decide) smallRow4957_covers

def smallNode4999 : Graph := .node 4999 smallRow4999 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode4999_covers : Graph.Covers PrimePairVertices.smallTree smallNode4999 :=
  Graph.node_covers _ 4999 smallRow4999 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow4999_covers

def smallNode5003 : Graph := .node 5003 smallRow5003 smallNode4999 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5003_covers : Graph.Covers PrimePairVertices.smallTree smallNode5003 :=
  Graph.node_covers _ 5003 smallRow5003 smallNode4999 smallEmpty
    smallNode4999_covers smallEmpty_covers (by decide) (by decide) smallRow5003_covers

def smallNode5011 : Graph := .node 5011 smallRow5011 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5011_covers : Graph.Covers PrimePairVertices.smallTree smallNode5011 :=
  Graph.node_covers _ 5011 smallRow5011 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5011_covers

def smallNode5009 : Graph := .node 5009 smallRow5009 smallNode5003 smallNode5011
set_option maxRecDepth 4096 in
theorem smallNode5009_covers : Graph.Covers PrimePairVertices.smallTree smallNode5009 :=
  Graph.node_covers _ 5009 smallRow5009 smallNode5003 smallNode5011
    smallNode5003_covers smallNode5011_covers (by decide) (by decide) smallRow5009_covers

def smallNode5023 : Graph := .node 5023 smallRow5023 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5023_covers : Graph.Covers PrimePairVertices.smallTree smallNode5023 :=
  Graph.node_covers _ 5023 smallRow5023 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5023_covers

def smallNode5039 : Graph := .node 5039 smallRow5039 smallNode5023 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5039_covers : Graph.Covers PrimePairVertices.smallTree smallNode5039 :=
  Graph.node_covers _ 5039 smallRow5039 smallNode5023 smallEmpty
    smallNode5023_covers smallEmpty_covers (by decide) (by decide) smallRow5039_covers

def smallNode5059 : Graph := .node 5059 smallRow5059 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5059_covers : Graph.Covers PrimePairVertices.smallTree smallNode5059 :=
  Graph.node_covers _ 5059 smallRow5059 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5059_covers

def smallNode5051 : Graph := .node 5051 smallRow5051 smallNode5039 smallNode5059
set_option maxRecDepth 4096 in
theorem smallNode5051_covers : Graph.Covers PrimePairVertices.smallTree smallNode5051 :=
  Graph.node_covers _ 5051 smallRow5051 smallNode5039 smallNode5059
    smallNode5039_covers smallNode5059_covers (by decide) (by decide) smallRow5051_covers

def smallNode5021 : Graph := .node 5021 smallRow5021 smallNode5009 smallNode5051
set_option maxRecDepth 4096 in
theorem smallNode5021_covers : Graph.Covers PrimePairVertices.smallTree smallNode5021 :=
  Graph.node_covers _ 5021 smallRow5021 smallNode5009 smallNode5051
    smallNode5009_covers smallNode5051_covers (by decide) (by decide) smallRow5021_covers

def smallNode4993 : Graph := .node 4993 smallRow4993 smallNode4957 smallNode5021
set_option maxRecDepth 4096 in
theorem smallNode4993_covers : Graph.Covers PrimePairVertices.smallTree smallNode4993 :=
  Graph.node_covers _ 4993 smallRow4993 smallNode4957 smallNode5021
    smallNode4957_covers smallNode5021_covers (by decide) (by decide) smallRow4993_covers

def smallNode4919 : Graph := .node 4919 smallRow4919 smallNode4801 smallNode4993
set_option maxRecDepth 4096 in
theorem smallNode4919_covers : Graph.Covers PrimePairVertices.smallTree smallNode4919 :=
  Graph.node_covers _ 4919 smallRow4919 smallNode4801 smallNode4993
    smallNode4801_covers smallNode4993_covers (by decide) (by decide) smallRow4919_covers

def smallNode5081 : Graph := .node 5081 smallRow5081 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5081_covers : Graph.Covers PrimePairVertices.smallTree smallNode5081 :=
  Graph.node_covers _ 5081 smallRow5081 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5081_covers

def smallNode5087 : Graph := .node 5087 smallRow5087 smallNode5081 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5087_covers : Graph.Covers PrimePairVertices.smallTree smallNode5087 :=
  Graph.node_covers _ 5087 smallRow5087 smallNode5081 smallEmpty
    smallNode5081_covers smallEmpty_covers (by decide) (by decide) smallRow5087_covers

def smallNode5101 : Graph := .node 5101 smallRow5101 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5101_covers : Graph.Covers PrimePairVertices.smallTree smallNode5101 :=
  Graph.node_covers _ 5101 smallRow5101 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5101_covers

def smallNode5107 : Graph := .node 5107 smallRow5107 smallNode5101 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5107_covers : Graph.Covers PrimePairVertices.smallTree smallNode5107 :=
  Graph.node_covers _ 5107 smallRow5107 smallNode5101 smallEmpty
    smallNode5101_covers smallEmpty_covers (by decide) (by decide) smallRow5107_covers

def smallNode5099 : Graph := .node 5099 smallRow5099 smallNode5087 smallNode5107
set_option maxRecDepth 4096 in
theorem smallNode5099_covers : Graph.Covers PrimePairVertices.smallTree smallNode5099 :=
  Graph.node_covers _ 5099 smallRow5099 smallNode5087 smallNode5107
    smallNode5087_covers smallNode5107_covers (by decide) (by decide) smallRow5099_covers

def smallNode5119 : Graph := .node 5119 smallRow5119 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5119_covers : Graph.Covers PrimePairVertices.smallTree smallNode5119 :=
  Graph.node_covers _ 5119 smallRow5119 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5119_covers

def smallNode5147 : Graph := .node 5147 smallRow5147 smallNode5119 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5147_covers : Graph.Covers PrimePairVertices.smallTree smallNode5147 :=
  Graph.node_covers _ 5147 smallRow5147 smallNode5119 smallEmpty
    smallNode5119_covers smallEmpty_covers (by decide) (by decide) smallRow5147_covers

def smallNode5167 : Graph := .node 5167 smallRow5167 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5167_covers : Graph.Covers PrimePairVertices.smallTree smallNode5167 :=
  Graph.node_covers _ 5167 smallRow5167 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5167_covers

def smallNode5153 : Graph := .node 5153 smallRow5153 smallNode5147 smallNode5167
set_option maxRecDepth 4096 in
theorem smallNode5153_covers : Graph.Covers PrimePairVertices.smallTree smallNode5153 :=
  Graph.node_covers _ 5153 smallRow5153 smallNode5147 smallNode5167
    smallNode5147_covers smallNode5167_covers (by decide) (by decide) smallRow5153_covers

def smallNode5113 : Graph := .node 5113 smallRow5113 smallNode5099 smallNode5153
set_option maxRecDepth 4096 in
theorem smallNode5113_covers : Graph.Covers PrimePairVertices.smallTree smallNode5113 :=
  Graph.node_covers _ 5113 smallRow5113 smallNode5099 smallNode5153
    smallNode5099_covers smallNode5153_covers (by decide) (by decide) smallRow5113_covers

def smallNode5179 : Graph := .node 5179 smallRow5179 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5179_covers : Graph.Covers PrimePairVertices.smallTree smallNode5179 :=
  Graph.node_covers _ 5179 smallRow5179 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5179_covers

def smallNode5189 : Graph := .node 5189 smallRow5189 smallNode5179 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5189_covers : Graph.Covers PrimePairVertices.smallTree smallNode5189 :=
  Graph.node_covers _ 5189 smallRow5189 smallNode5179 smallEmpty
    smallNode5179_covers smallEmpty_covers (by decide) (by decide) smallRow5189_covers

def smallNode5209 : Graph := .node 5209 smallRow5209 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5209_covers : Graph.Covers PrimePairVertices.smallTree smallNode5209 :=
  Graph.node_covers _ 5209 smallRow5209 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5209_covers

def smallNode5197 : Graph := .node 5197 smallRow5197 smallNode5189 smallNode5209
set_option maxRecDepth 4096 in
theorem smallNode5197_covers : Graph.Covers PrimePairVertices.smallTree smallNode5197 :=
  Graph.node_covers _ 5197 smallRow5197 smallNode5189 smallNode5209
    smallNode5189_covers smallNode5209_covers (by decide) (by decide) smallRow5197_covers

def smallNode5231 : Graph := .node 5231 smallRow5231 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5231_covers : Graph.Covers PrimePairVertices.smallTree smallNode5231 :=
  Graph.node_covers _ 5231 smallRow5231 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5231_covers

def smallNode5233 : Graph := .node 5233 smallRow5233 smallNode5231 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5233_covers : Graph.Covers PrimePairVertices.smallTree smallNode5233 :=
  Graph.node_covers _ 5233 smallRow5233 smallNode5231 smallEmpty
    smallNode5231_covers smallEmpty_covers (by decide) (by decide) smallRow5233_covers

def smallNode5261 : Graph := .node 5261 smallRow5261 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5261_covers : Graph.Covers PrimePairVertices.smallTree smallNode5261 :=
  Graph.node_covers _ 5261 smallRow5261 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5261_covers

def smallNode5237 : Graph := .node 5237 smallRow5237 smallNode5233 smallNode5261
set_option maxRecDepth 4096 in
theorem smallNode5237_covers : Graph.Covers PrimePairVertices.smallTree smallNode5237 :=
  Graph.node_covers _ 5237 smallRow5237 smallNode5233 smallNode5261
    smallNode5233_covers smallNode5261_covers (by decide) (by decide) smallRow5237_covers

def smallNode5227 : Graph := .node 5227 smallRow5227 smallNode5197 smallNode5237
set_option maxRecDepth 4096 in
theorem smallNode5227_covers : Graph.Covers PrimePairVertices.smallTree smallNode5227 :=
  Graph.node_covers _ 5227 smallRow5227 smallNode5197 smallNode5237
    smallNode5197_covers smallNode5237_covers (by decide) (by decide) smallRow5227_covers

def smallNode5171 : Graph := .node 5171 smallRow5171 smallNode5113 smallNode5227
set_option maxRecDepth 4096 in
theorem smallNode5171_covers : Graph.Covers PrimePairVertices.smallTree smallNode5171 :=
  Graph.node_covers _ 5171 smallRow5171 smallNode5113 smallNode5227
    smallNode5113_covers smallNode5227_covers (by decide) (by decide) smallRow5171_covers

def smallNode5279 : Graph := .node 5279 smallRow5279 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5279_covers : Graph.Covers PrimePairVertices.smallTree smallNode5279 :=
  Graph.node_covers _ 5279 smallRow5279 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5279_covers

def smallNode5281 : Graph := .node 5281 smallRow5281 smallNode5279 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5281_covers : Graph.Covers PrimePairVertices.smallTree smallNode5281 :=
  Graph.node_covers _ 5281 smallRow5281 smallNode5279 smallEmpty
    smallNode5279_covers smallEmpty_covers (by decide) (by decide) smallRow5281_covers

def smallNode5303 : Graph := .node 5303 smallRow5303 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5303_covers : Graph.Covers PrimePairVertices.smallTree smallNode5303 :=
  Graph.node_covers _ 5303 smallRow5303 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5303_covers

def smallNode5297 : Graph := .node 5297 smallRow5297 smallNode5281 smallNode5303
set_option maxRecDepth 4096 in
theorem smallNode5297_covers : Graph.Covers PrimePairVertices.smallTree smallNode5297 :=
  Graph.node_covers _ 5297 smallRow5297 smallNode5281 smallNode5303
    smallNode5281_covers smallNode5303_covers (by decide) (by decide) smallRow5297_covers

def smallNode5323 : Graph := .node 5323 smallRow5323 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5323_covers : Graph.Covers PrimePairVertices.smallTree smallNode5323 :=
  Graph.node_covers _ 5323 smallRow5323 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5323_covers

def smallNode5333 : Graph := .node 5333 smallRow5333 smallNode5323 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5333_covers : Graph.Covers PrimePairVertices.smallTree smallNode5333 :=
  Graph.node_covers _ 5333 smallRow5333 smallNode5323 smallEmpty
    smallNode5323_covers smallEmpty_covers (by decide) (by decide) smallRow5333_covers

def smallNode5351 : Graph := .node 5351 smallRow5351 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5351_covers : Graph.Covers PrimePairVertices.smallTree smallNode5351 :=
  Graph.node_covers _ 5351 smallRow5351 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5351_covers

def smallNode5347 : Graph := .node 5347 smallRow5347 smallNode5333 smallNode5351
set_option maxRecDepth 4096 in
theorem smallNode5347_covers : Graph.Covers PrimePairVertices.smallTree smallNode5347 :=
  Graph.node_covers _ 5347 smallRow5347 smallNode5333 smallNode5351
    smallNode5333_covers smallNode5351_covers (by decide) (by decide) smallRow5347_covers

def smallNode5309 : Graph := .node 5309 smallRow5309 smallNode5297 smallNode5347
set_option maxRecDepth 4096 in
theorem smallNode5309_covers : Graph.Covers PrimePairVertices.smallTree smallNode5309 :=
  Graph.node_covers _ 5309 smallRow5309 smallNode5297 smallNode5347
    smallNode5297_covers smallNode5347_covers (by decide) (by decide) smallRow5309_covers

def smallNode5387 : Graph := .node 5387 smallRow5387 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5387_covers : Graph.Covers PrimePairVertices.smallTree smallNode5387 :=
  Graph.node_covers _ 5387 smallRow5387 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5387_covers

def smallNode5393 : Graph := .node 5393 smallRow5393 smallNode5387 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5393_covers : Graph.Covers PrimePairVertices.smallTree smallNode5393 :=
  Graph.node_covers _ 5393 smallRow5393 smallNode5387 smallEmpty
    smallNode5387_covers smallEmpty_covers (by decide) (by decide) smallRow5393_covers

def smallNode5407 : Graph := .node 5407 smallRow5407 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5407_covers : Graph.Covers PrimePairVertices.smallTree smallNode5407 :=
  Graph.node_covers _ 5407 smallRow5407 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5407_covers

def smallNode5399 : Graph := .node 5399 smallRow5399 smallNode5393 smallNode5407
set_option maxRecDepth 4096 in
theorem smallNode5399_covers : Graph.Covers PrimePairVertices.smallTree smallNode5399 :=
  Graph.node_covers _ 5399 smallRow5399 smallNode5393 smallNode5407
    smallNode5393_covers smallNode5407_covers (by decide) (by decide) smallRow5399_covers

def smallNode5417 : Graph := .node 5417 smallRow5417 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5417_covers : Graph.Covers PrimePairVertices.smallTree smallNode5417 :=
  Graph.node_covers _ 5417 smallRow5417 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5417_covers

def smallNode5419 : Graph := .node 5419 smallRow5419 smallNode5417 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5419_covers : Graph.Covers PrimePairVertices.smallTree smallNode5419 :=
  Graph.node_covers _ 5419 smallRow5419 smallNode5417 smallEmpty
    smallNode5417_covers smallEmpty_covers (by decide) (by decide) smallRow5419_covers

def smallNode5437 : Graph := .node 5437 smallRow5437 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5437_covers : Graph.Covers PrimePairVertices.smallTree smallNode5437 :=
  Graph.node_covers _ 5437 smallRow5437 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5437_covers

def smallNode5431 : Graph := .node 5431 smallRow5431 smallNode5419 smallNode5437
set_option maxRecDepth 4096 in
theorem smallNode5431_covers : Graph.Covers PrimePairVertices.smallTree smallNode5431 :=
  Graph.node_covers _ 5431 smallRow5431 smallNode5419 smallNode5437
    smallNode5419_covers smallNode5437_covers (by decide) (by decide) smallRow5431_covers

def smallNode5413 : Graph := .node 5413 smallRow5413 smallNode5399 smallNode5431
set_option maxRecDepth 4096 in
theorem smallNode5413_covers : Graph.Covers PrimePairVertices.smallTree smallNode5413 :=
  Graph.node_covers _ 5413 smallRow5413 smallNode5399 smallNode5431
    smallNode5399_covers smallNode5431_covers (by decide) (by decide) smallRow5413_covers

def smallNode5381 : Graph := .node 5381 smallRow5381 smallNode5309 smallNode5413
set_option maxRecDepth 4096 in
theorem smallNode5381_covers : Graph.Covers PrimePairVertices.smallTree smallNode5381 :=
  Graph.node_covers _ 5381 smallRow5381 smallNode5309 smallNode5413
    smallNode5309_covers smallNode5413_covers (by decide) (by decide) smallRow5381_covers

def smallNode5273 : Graph := .node 5273 smallRow5273 smallNode5171 smallNode5381
set_option maxRecDepth 4096 in
theorem smallNode5273_covers : Graph.Covers PrimePairVertices.smallTree smallNode5273 :=
  Graph.node_covers _ 5273 smallRow5273 smallNode5171 smallNode5381
    smallNode5171_covers smallNode5381_covers (by decide) (by decide) smallRow5273_covers

def smallNode5077 : Graph := .node 5077 smallRow5077 smallNode4919 smallNode5273
set_option maxRecDepth 4096 in
theorem smallNode5077_covers : Graph.Covers PrimePairVertices.smallTree smallNode5077 :=
  Graph.node_covers _ 5077 smallRow5077 smallNode4919 smallNode5273
    smallNode4919_covers smallNode5273_covers (by decide) (by decide) smallRow5077_covers

def smallNode4721 : Graph := .node 4721 smallRow4721 smallNode4357 smallNode5077
set_option maxRecDepth 4096 in
theorem smallNode4721_covers : Graph.Covers PrimePairVertices.smallTree smallNode4721 :=
  Graph.node_covers _ 4721 smallRow4721 smallNode4357 smallNode5077
    smallNode4357_covers smallNode5077_covers (by decide) (by decide) smallRow4721_covers

def smallNode4007 : Graph := .node 4007 smallRow4007 smallNode3331 smallNode4721
set_option maxRecDepth 4096 in
theorem smallNode4007_covers : Graph.Covers PrimePairVertices.smallTree smallNode4007 :=
  Graph.node_covers _ 4007 smallRow4007 smallNode3331 smallNode4721
    smallNode3331_covers smallNode4721_covers (by decide) (by decide) smallRow4007_covers

def smallNode5443 : Graph := .node 5443 smallRow5443 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5443_covers : Graph.Covers PrimePairVertices.smallTree smallNode5443 :=
  Graph.node_covers _ 5443 smallRow5443 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5443_covers

def smallNode5449 : Graph := .node 5449 smallRow5449 smallNode5443 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5449_covers : Graph.Covers PrimePairVertices.smallTree smallNode5449 :=
  Graph.node_covers _ 5449 smallRow5449 smallNode5443 smallEmpty
    smallNode5443_covers smallEmpty_covers (by decide) (by decide) smallRow5449_covers

def smallNode5477 : Graph := .node 5477 smallRow5477 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5477_covers : Graph.Covers PrimePairVertices.smallTree smallNode5477 :=
  Graph.node_covers _ 5477 smallRow5477 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5477_covers

def smallNode5479 : Graph := .node 5479 smallRow5479 smallNode5477 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5479_covers : Graph.Covers PrimePairVertices.smallTree smallNode5479 :=
  Graph.node_covers _ 5479 smallRow5479 smallNode5477 smallEmpty
    smallNode5477_covers smallEmpty_covers (by decide) (by decide) smallRow5479_covers

def smallNode5471 : Graph := .node 5471 smallRow5471 smallNode5449 smallNode5479
set_option maxRecDepth 4096 in
theorem smallNode5471_covers : Graph.Covers PrimePairVertices.smallTree smallNode5471 :=
  Graph.node_covers _ 5471 smallRow5471 smallNode5449 smallNode5479
    smallNode5449_covers smallNode5479_covers (by decide) (by decide) smallRow5471_covers

def smallNode5501 : Graph := .node 5501 smallRow5501 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5501_covers : Graph.Covers PrimePairVertices.smallTree smallNode5501 :=
  Graph.node_covers _ 5501 smallRow5501 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5501_covers

def smallNode5503 : Graph := .node 5503 smallRow5503 smallNode5501 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5503_covers : Graph.Covers PrimePairVertices.smallTree smallNode5503 :=
  Graph.node_covers _ 5503 smallRow5503 smallNode5501 smallEmpty
    smallNode5501_covers smallEmpty_covers (by decide) (by decide) smallRow5503_covers

def smallNode5519 : Graph := .node 5519 smallRow5519 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5519_covers : Graph.Covers PrimePairVertices.smallTree smallNode5519 :=
  Graph.node_covers _ 5519 smallRow5519 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5519_covers

def smallNode5507 : Graph := .node 5507 smallRow5507 smallNode5503 smallNode5519
set_option maxRecDepth 4096 in
theorem smallNode5507_covers : Graph.Covers PrimePairVertices.smallTree smallNode5507 :=
  Graph.node_covers _ 5507 smallRow5507 smallNode5503 smallNode5519
    smallNode5503_covers smallNode5519_covers (by decide) (by decide) smallRow5507_covers

def smallNode5483 : Graph := .node 5483 smallRow5483 smallNode5471 smallNode5507
set_option maxRecDepth 4096 in
theorem smallNode5483_covers : Graph.Covers PrimePairVertices.smallTree smallNode5483 :=
  Graph.node_covers _ 5483 smallRow5483 smallNode5471 smallNode5507
    smallNode5471_covers smallNode5507_covers (by decide) (by decide) smallRow5483_covers

def smallNode5527 : Graph := .node 5527 smallRow5527 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5527_covers : Graph.Covers PrimePairVertices.smallTree smallNode5527 :=
  Graph.node_covers _ 5527 smallRow5527 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5527_covers

def smallNode5531 : Graph := .node 5531 smallRow5531 smallNode5527 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5531_covers : Graph.Covers PrimePairVertices.smallTree smallNode5531 :=
  Graph.node_covers _ 5531 smallRow5531 smallNode5527 smallEmpty
    smallNode5527_covers smallEmpty_covers (by decide) (by decide) smallRow5531_covers

def smallNode5563 : Graph := .node 5563 smallRow5563 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5563_covers : Graph.Covers PrimePairVertices.smallTree smallNode5563 :=
  Graph.node_covers _ 5563 smallRow5563 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5563_covers

def smallNode5557 : Graph := .node 5557 smallRow5557 smallNode5531 smallNode5563
set_option maxRecDepth 4096 in
theorem smallNode5557_covers : Graph.Covers PrimePairVertices.smallTree smallNode5557 :=
  Graph.node_covers _ 5557 smallRow5557 smallNode5531 smallNode5563
    smallNode5531_covers smallNode5563_covers (by decide) (by decide) smallRow5557_covers

def smallNode5573 : Graph := .node 5573 smallRow5573 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5573_covers : Graph.Covers PrimePairVertices.smallTree smallNode5573 :=
  Graph.node_covers _ 5573 smallRow5573 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5573_covers

def smallNode5581 : Graph := .node 5581 smallRow5581 smallNode5573 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5581_covers : Graph.Covers PrimePairVertices.smallTree smallNode5581 :=
  Graph.node_covers _ 5581 smallRow5581 smallNode5573 smallEmpty
    smallNode5573_covers smallEmpty_covers (by decide) (by decide) smallRow5581_covers

def smallNode5623 : Graph := .node 5623 smallRow5623 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5623_covers : Graph.Covers PrimePairVertices.smallTree smallNode5623 :=
  Graph.node_covers _ 5623 smallRow5623 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5623_covers

def smallNode5591 : Graph := .node 5591 smallRow5591 smallNode5581 smallNode5623
set_option maxRecDepth 4096 in
theorem smallNode5591_covers : Graph.Covers PrimePairVertices.smallTree smallNode5591 :=
  Graph.node_covers _ 5591 smallRow5591 smallNode5581 smallNode5623
    smallNode5581_covers smallNode5623_covers (by decide) (by decide) smallRow5591_covers

def smallNode5569 : Graph := .node 5569 smallRow5569 smallNode5557 smallNode5591
set_option maxRecDepth 4096 in
theorem smallNode5569_covers : Graph.Covers PrimePairVertices.smallTree smallNode5569 :=
  Graph.node_covers _ 5569 smallRow5569 smallNode5557 smallNode5591
    smallNode5557_covers smallNode5591_covers (by decide) (by decide) smallRow5569_covers

def smallNode5521 : Graph := .node 5521 smallRow5521 smallNode5483 smallNode5569
set_option maxRecDepth 4096 in
theorem smallNode5521_covers : Graph.Covers PrimePairVertices.smallTree smallNode5521 :=
  Graph.node_covers _ 5521 smallRow5521 smallNode5483 smallNode5569
    smallNode5483_covers smallNode5569_covers (by decide) (by decide) smallRow5521_covers

def smallNode5641 : Graph := .node 5641 smallRow5641 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5641_covers : Graph.Covers PrimePairVertices.smallTree smallNode5641 :=
  Graph.node_covers _ 5641 smallRow5641 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5641_covers

def smallNode5647 : Graph := .node 5647 smallRow5647 smallNode5641 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5647_covers : Graph.Covers PrimePairVertices.smallTree smallNode5647 :=
  Graph.node_covers _ 5647 smallRow5647 smallNode5641 smallEmpty
    smallNode5641_covers smallEmpty_covers (by decide) (by decide) smallRow5647_covers

def smallNode5653 : Graph := .node 5653 smallRow5653 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5653_covers : Graph.Covers PrimePairVertices.smallTree smallNode5653 :=
  Graph.node_covers _ 5653 smallRow5653 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5653_covers

def smallNode5657 : Graph := .node 5657 smallRow5657 smallNode5653 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5657_covers : Graph.Covers PrimePairVertices.smallTree smallNode5657 :=
  Graph.node_covers _ 5657 smallRow5657 smallNode5653 smallEmpty
    smallNode5653_covers smallEmpty_covers (by decide) (by decide) smallRow5657_covers

def smallNode5651 : Graph := .node 5651 smallRow5651 smallNode5647 smallNode5657
set_option maxRecDepth 4096 in
theorem smallNode5651_covers : Graph.Covers PrimePairVertices.smallTree smallNode5651 :=
  Graph.node_covers _ 5651 smallRow5651 smallNode5647 smallNode5657
    smallNode5647_covers smallNode5657_covers (by decide) (by decide) smallRow5651_covers

def smallNode5669 : Graph := .node 5669 smallRow5669 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5669_covers : Graph.Covers PrimePairVertices.smallTree smallNode5669 :=
  Graph.node_covers _ 5669 smallRow5669 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5669_covers

def smallNode5683 : Graph := .node 5683 smallRow5683 smallNode5669 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5683_covers : Graph.Covers PrimePairVertices.smallTree smallNode5683 :=
  Graph.node_covers _ 5683 smallRow5683 smallNode5669 smallEmpty
    smallNode5669_covers smallEmpty_covers (by decide) (by decide) smallRow5683_covers

def smallNode5693 : Graph := .node 5693 smallRow5693 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5693_covers : Graph.Covers PrimePairVertices.smallTree smallNode5693 :=
  Graph.node_covers _ 5693 smallRow5693 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5693_covers

def smallNode5689 : Graph := .node 5689 smallRow5689 smallNode5683 smallNode5693
set_option maxRecDepth 4096 in
theorem smallNode5689_covers : Graph.Covers PrimePairVertices.smallTree smallNode5689 :=
  Graph.node_covers _ 5689 smallRow5689 smallNode5683 smallNode5693
    smallNode5683_covers smallNode5693_covers (by decide) (by decide) smallRow5689_covers

def smallNode5659 : Graph := .node 5659 smallRow5659 smallNode5651 smallNode5689
set_option maxRecDepth 4096 in
theorem smallNode5659_covers : Graph.Covers PrimePairVertices.smallTree smallNode5659 :=
  Graph.node_covers _ 5659 smallRow5659 smallNode5651 smallNode5689
    smallNode5651_covers smallNode5689_covers (by decide) (by decide) smallRow5659_covers

def smallNode5711 : Graph := .node 5711 smallRow5711 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5711_covers : Graph.Covers PrimePairVertices.smallTree smallNode5711 :=
  Graph.node_covers _ 5711 smallRow5711 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5711_covers

def smallNode5717 : Graph := .node 5717 smallRow5717 smallNode5711 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5717_covers : Graph.Covers PrimePairVertices.smallTree smallNode5717 :=
  Graph.node_covers _ 5717 smallRow5717 smallNode5711 smallEmpty
    smallNode5711_covers smallEmpty_covers (by decide) (by decide) smallRow5717_covers

def smallNode5741 : Graph := .node 5741 smallRow5741 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5741_covers : Graph.Covers PrimePairVertices.smallTree smallNode5741 :=
  Graph.node_covers _ 5741 smallRow5741 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5741_covers

def smallNode5737 : Graph := .node 5737 smallRow5737 smallNode5717 smallNode5741
set_option maxRecDepth 4096 in
theorem smallNode5737_covers : Graph.Covers PrimePairVertices.smallTree smallNode5737 :=
  Graph.node_covers _ 5737 smallRow5737 smallNode5717 smallNode5741
    smallNode5717_covers smallNode5741_covers (by decide) (by decide) smallRow5737_covers

def smallNode5749 : Graph := .node 5749 smallRow5749 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5749_covers : Graph.Covers PrimePairVertices.smallTree smallNode5749 :=
  Graph.node_covers _ 5749 smallRow5749 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5749_covers

def smallNode5779 : Graph := .node 5779 smallRow5779 smallNode5749 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5779_covers : Graph.Covers PrimePairVertices.smallTree smallNode5779 :=
  Graph.node_covers _ 5779 smallRow5779 smallNode5749 smallEmpty
    smallNode5749_covers smallEmpty_covers (by decide) (by decide) smallRow5779_covers

def smallNode5791 : Graph := .node 5791 smallRow5791 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5791_covers : Graph.Covers PrimePairVertices.smallTree smallNode5791 :=
  Graph.node_covers _ 5791 smallRow5791 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5791_covers

def smallNode5783 : Graph := .node 5783 smallRow5783 smallNode5779 smallNode5791
set_option maxRecDepth 4096 in
theorem smallNode5783_covers : Graph.Covers PrimePairVertices.smallTree smallNode5783 :=
  Graph.node_covers _ 5783 smallRow5783 smallNode5779 smallNode5791
    smallNode5779_covers smallNode5791_covers (by decide) (by decide) smallRow5783_covers

def smallNode5743 : Graph := .node 5743 smallRow5743 smallNode5737 smallNode5783
set_option maxRecDepth 4096 in
theorem smallNode5743_covers : Graph.Covers PrimePairVertices.smallTree smallNode5743 :=
  Graph.node_covers _ 5743 smallRow5743 smallNode5737 smallNode5783
    smallNode5737_covers smallNode5783_covers (by decide) (by decide) smallRow5743_covers

def smallNode5701 : Graph := .node 5701 smallRow5701 smallNode5659 smallNode5743
set_option maxRecDepth 4096 in
theorem smallNode5701_covers : Graph.Covers PrimePairVertices.smallTree smallNode5701 :=
  Graph.node_covers _ 5701 smallRow5701 smallNode5659 smallNode5743
    smallNode5659_covers smallNode5743_covers (by decide) (by decide) smallRow5701_covers

def smallNode5639 : Graph := .node 5639 smallRow5639 smallNode5521 smallNode5701
set_option maxRecDepth 4096 in
theorem smallNode5639_covers : Graph.Covers PrimePairVertices.smallTree smallNode5639 :=
  Graph.node_covers _ 5639 smallRow5639 smallNode5521 smallNode5701
    smallNode5521_covers smallNode5701_covers (by decide) (by decide) smallRow5639_covers

def smallNode5807 : Graph := .node 5807 smallRow5807 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5807_covers : Graph.Covers PrimePairVertices.smallTree smallNode5807 :=
  Graph.node_covers _ 5807 smallRow5807 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5807_covers

def smallNode5813 : Graph := .node 5813 smallRow5813 smallNode5807 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5813_covers : Graph.Covers PrimePairVertices.smallTree smallNode5813 :=
  Graph.node_covers _ 5813 smallRow5813 smallNode5807 smallEmpty
    smallNode5807_covers smallEmpty_covers (by decide) (by decide) smallRow5813_covers

def smallNode5827 : Graph := .node 5827 smallRow5827 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5827_covers : Graph.Covers PrimePairVertices.smallTree smallNode5827 :=
  Graph.node_covers _ 5827 smallRow5827 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5827_covers

def smallNode5839 : Graph := .node 5839 smallRow5839 smallNode5827 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5839_covers : Graph.Covers PrimePairVertices.smallTree smallNode5839 :=
  Graph.node_covers _ 5839 smallRow5839 smallNode5827 smallEmpty
    smallNode5827_covers smallEmpty_covers (by decide) (by decide) smallRow5839_covers

def smallNode5821 : Graph := .node 5821 smallRow5821 smallNode5813 smallNode5839
set_option maxRecDepth 4096 in
theorem smallNode5821_covers : Graph.Covers PrimePairVertices.smallTree smallNode5821 :=
  Graph.node_covers _ 5821 smallRow5821 smallNode5813 smallNode5839
    smallNode5813_covers smallNode5839_covers (by decide) (by decide) smallRow5821_covers

def smallNode5849 : Graph := .node 5849 smallRow5849 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5849_covers : Graph.Covers PrimePairVertices.smallTree smallNode5849 :=
  Graph.node_covers _ 5849 smallRow5849 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5849_covers

def smallNode5851 : Graph := .node 5851 smallRow5851 smallNode5849 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5851_covers : Graph.Covers PrimePairVertices.smallTree smallNode5851 :=
  Graph.node_covers _ 5851 smallRow5851 smallNode5849 smallEmpty
    smallNode5849_covers smallEmpty_covers (by decide) (by decide) smallRow5851_covers

def smallNode5861 : Graph := .node 5861 smallRow5861 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5861_covers : Graph.Covers PrimePairVertices.smallTree smallNode5861 :=
  Graph.node_covers _ 5861 smallRow5861 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5861_covers

def smallNode5857 : Graph := .node 5857 smallRow5857 smallNode5851 smallNode5861
set_option maxRecDepth 4096 in
theorem smallNode5857_covers : Graph.Covers PrimePairVertices.smallTree smallNode5857 :=
  Graph.node_covers _ 5857 smallRow5857 smallNode5851 smallNode5861
    smallNode5851_covers smallNode5861_covers (by decide) (by decide) smallRow5857_covers

def smallNode5843 : Graph := .node 5843 smallRow5843 smallNode5821 smallNode5857
set_option maxRecDepth 4096 in
theorem smallNode5843_covers : Graph.Covers PrimePairVertices.smallTree smallNode5843 :=
  Graph.node_covers _ 5843 smallRow5843 smallNode5821 smallNode5857
    smallNode5821_covers smallNode5857_covers (by decide) (by decide) smallRow5843_covers

def smallNode5869 : Graph := .node 5869 smallRow5869 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5869_covers : Graph.Covers PrimePairVertices.smallTree smallNode5869 :=
  Graph.node_covers _ 5869 smallRow5869 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5869_covers

def smallNode5879 : Graph := .node 5879 smallRow5879 smallNode5869 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5879_covers : Graph.Covers PrimePairVertices.smallTree smallNode5879 :=
  Graph.node_covers _ 5879 smallRow5879 smallNode5869 smallEmpty
    smallNode5869_covers smallEmpty_covers (by decide) (by decide) smallRow5879_covers

def smallNode5897 : Graph := .node 5897 smallRow5897 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5897_covers : Graph.Covers PrimePairVertices.smallTree smallNode5897 :=
  Graph.node_covers _ 5897 smallRow5897 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5897_covers

def smallNode5881 : Graph := .node 5881 smallRow5881 smallNode5879 smallNode5897
set_option maxRecDepth 4096 in
theorem smallNode5881_covers : Graph.Covers PrimePairVertices.smallTree smallNode5881 :=
  Graph.node_covers _ 5881 smallRow5881 smallNode5879 smallNode5897
    smallNode5879_covers smallNode5897_covers (by decide) (by decide) smallRow5881_covers

def smallNode5923 : Graph := .node 5923 smallRow5923 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5923_covers : Graph.Covers PrimePairVertices.smallTree smallNode5923 :=
  Graph.node_covers _ 5923 smallRow5923 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5923_covers

def smallNode5927 : Graph := .node 5927 smallRow5927 smallNode5923 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5927_covers : Graph.Covers PrimePairVertices.smallTree smallNode5927 :=
  Graph.node_covers _ 5927 smallRow5927 smallNode5923 smallEmpty
    smallNode5923_covers smallEmpty_covers (by decide) (by decide) smallRow5927_covers

def smallNode5953 : Graph := .node 5953 smallRow5953 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5953_covers : Graph.Covers PrimePairVertices.smallTree smallNode5953 :=
  Graph.node_covers _ 5953 smallRow5953 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5953_covers

def smallNode5939 : Graph := .node 5939 smallRow5939 smallNode5927 smallNode5953
set_option maxRecDepth 4096 in
theorem smallNode5939_covers : Graph.Covers PrimePairVertices.smallTree smallNode5939 :=
  Graph.node_covers _ 5939 smallRow5939 smallNode5927 smallNode5953
    smallNode5927_covers smallNode5953_covers (by decide) (by decide) smallRow5939_covers

def smallNode5903 : Graph := .node 5903 smallRow5903 smallNode5881 smallNode5939
set_option maxRecDepth 4096 in
theorem smallNode5903_covers : Graph.Covers PrimePairVertices.smallTree smallNode5903 :=
  Graph.node_covers _ 5903 smallRow5903 smallNode5881 smallNode5939
    smallNode5881_covers smallNode5939_covers (by decide) (by decide) smallRow5903_covers

def smallNode5867 : Graph := .node 5867 smallRow5867 smallNode5843 smallNode5903
set_option maxRecDepth 4096 in
theorem smallNode5867_covers : Graph.Covers PrimePairVertices.smallTree smallNode5867 :=
  Graph.node_covers _ 5867 smallRow5867 smallNode5843 smallNode5903
    smallNode5843_covers smallNode5903_covers (by decide) (by decide) smallRow5867_covers

def smallNode5987 : Graph := .node 5987 smallRow5987 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode5987_covers : Graph.Covers PrimePairVertices.smallTree smallNode5987 :=
  Graph.node_covers _ 5987 smallRow5987 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow5987_covers

def smallNode6007 : Graph := .node 6007 smallRow6007 smallNode5987 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6007_covers : Graph.Covers PrimePairVertices.smallTree smallNode6007 :=
  Graph.node_covers _ 6007 smallRow6007 smallNode5987 smallEmpty
    smallNode5987_covers smallEmpty_covers (by decide) (by decide) smallRow6007_covers

def smallNode6029 : Graph := .node 6029 smallRow6029 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6029_covers : Graph.Covers PrimePairVertices.smallTree smallNode6029 :=
  Graph.node_covers _ 6029 smallRow6029 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6029_covers

def smallNode6011 : Graph := .node 6011 smallRow6011 smallNode6007 smallNode6029
set_option maxRecDepth 4096 in
theorem smallNode6011_covers : Graph.Covers PrimePairVertices.smallTree smallNode6011 :=
  Graph.node_covers _ 6011 smallRow6011 smallNode6007 smallNode6029
    smallNode6007_covers smallNode6029_covers (by decide) (by decide) smallRow6011_covers

def smallNode6043 : Graph := .node 6043 smallRow6043 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6043_covers : Graph.Covers PrimePairVertices.smallTree smallNode6043 :=
  Graph.node_covers _ 6043 smallRow6043 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6043_covers

def smallNode6047 : Graph := .node 6047 smallRow6047 smallNode6043 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6047_covers : Graph.Covers PrimePairVertices.smallTree smallNode6047 :=
  Graph.node_covers _ 6047 smallRow6047 smallNode6043 smallEmpty
    smallNode6043_covers smallEmpty_covers (by decide) (by decide) smallRow6047_covers

def smallNode6067 : Graph := .node 6067 smallRow6067 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6067_covers : Graph.Covers PrimePairVertices.smallTree smallNode6067 :=
  Graph.node_covers _ 6067 smallRow6067 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6067_covers

def smallNode6053 : Graph := .node 6053 smallRow6053 smallNode6047 smallNode6067
set_option maxRecDepth 4096 in
theorem smallNode6053_covers : Graph.Covers PrimePairVertices.smallTree smallNode6053 :=
  Graph.node_covers _ 6053 smallRow6053 smallNode6047 smallNode6067
    smallNode6047_covers smallNode6067_covers (by decide) (by decide) smallRow6053_covers

def smallNode6037 : Graph := .node 6037 smallRow6037 smallNode6011 smallNode6053
set_option maxRecDepth 4096 in
theorem smallNode6037_covers : Graph.Covers PrimePairVertices.smallTree smallNode6037 :=
  Graph.node_covers _ 6037 smallRow6037 smallNode6011 smallNode6053
    smallNode6011_covers smallNode6053_covers (by decide) (by decide) smallRow6037_covers

def smallNode6079 : Graph := .node 6079 smallRow6079 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6079_covers : Graph.Covers PrimePairVertices.smallTree smallNode6079 :=
  Graph.node_covers _ 6079 smallRow6079 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6079_covers

def smallNode6089 : Graph := .node 6089 smallRow6089 smallNode6079 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6089_covers : Graph.Covers PrimePairVertices.smallTree smallNode6089 :=
  Graph.node_covers _ 6089 smallRow6089 smallNode6079 smallEmpty
    smallNode6079_covers smallEmpty_covers (by decide) (by decide) smallRow6089_covers

def smallNode6101 : Graph := .node 6101 smallRow6101 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6101_covers : Graph.Covers PrimePairVertices.smallTree smallNode6101 :=
  Graph.node_covers _ 6101 smallRow6101 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6101_covers

def smallNode6091 : Graph := .node 6091 smallRow6091 smallNode6089 smallNode6101
set_option maxRecDepth 4096 in
theorem smallNode6091_covers : Graph.Covers PrimePairVertices.smallTree smallNode6091 :=
  Graph.node_covers _ 6091 smallRow6091 smallNode6089 smallNode6101
    smallNode6089_covers smallNode6101_covers (by decide) (by decide) smallRow6091_covers

def smallNode6121 : Graph := .node 6121 smallRow6121 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6121_covers : Graph.Covers PrimePairVertices.smallTree smallNode6121 :=
  Graph.node_covers _ 6121 smallRow6121 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6121_covers

def smallNode6131 : Graph := .node 6131 smallRow6131 smallNode6121 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6131_covers : Graph.Covers PrimePairVertices.smallTree smallNode6131 :=
  Graph.node_covers _ 6131 smallRow6131 smallNode6121 smallEmpty
    smallNode6121_covers smallEmpty_covers (by decide) (by decide) smallRow6131_covers

def smallNode6143 : Graph := .node 6143 smallRow6143 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6143_covers : Graph.Covers PrimePairVertices.smallTree smallNode6143 :=
  Graph.node_covers _ 6143 smallRow6143 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6143_covers

def smallNode6133 : Graph := .node 6133 smallRow6133 smallNode6131 smallNode6143
set_option maxRecDepth 4096 in
theorem smallNode6133_covers : Graph.Covers PrimePairVertices.smallTree smallNode6133 :=
  Graph.node_covers _ 6133 smallRow6133 smallNode6131 smallNode6143
    smallNode6131_covers smallNode6143_covers (by decide) (by decide) smallRow6133_covers

def smallNode6113 : Graph := .node 6113 smallRow6113 smallNode6091 smallNode6133
set_option maxRecDepth 4096 in
theorem smallNode6113_covers : Graph.Covers PrimePairVertices.smallTree smallNode6113 :=
  Graph.node_covers _ 6113 smallRow6113 smallNode6091 smallNode6133
    smallNode6091_covers smallNode6133_covers (by decide) (by decide) smallRow6113_covers

def smallNode6073 : Graph := .node 6073 smallRow6073 smallNode6037 smallNode6113
set_option maxRecDepth 4096 in
theorem smallNode6073_covers : Graph.Covers PrimePairVertices.smallTree smallNode6073 :=
  Graph.node_covers _ 6073 smallRow6073 smallNode6037 smallNode6113
    smallNode6037_covers smallNode6113_covers (by decide) (by decide) smallRow6073_covers

def smallNode5981 : Graph := .node 5981 smallRow5981 smallNode5867 smallNode6073
set_option maxRecDepth 4096 in
theorem smallNode5981_covers : Graph.Covers PrimePairVertices.smallTree smallNode5981 :=
  Graph.node_covers _ 5981 smallRow5981 smallNode5867 smallNode6073
    smallNode5867_covers smallNode6073_covers (by decide) (by decide) smallRow5981_covers

def smallNode5801 : Graph := .node 5801 smallRow5801 smallNode5639 smallNode5981
set_option maxRecDepth 4096 in
theorem smallNode5801_covers : Graph.Covers PrimePairVertices.smallTree smallNode5801 :=
  Graph.node_covers _ 5801 smallRow5801 smallNode5639 smallNode5981
    smallNode5639_covers smallNode5981_covers (by decide) (by decide) smallRow5801_covers

def smallNode6163 : Graph := .node 6163 smallRow6163 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6163_covers : Graph.Covers PrimePairVertices.smallTree smallNode6163 :=
  Graph.node_covers _ 6163 smallRow6163 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6163_covers

def smallNode6173 : Graph := .node 6173 smallRow6173 smallNode6163 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6173_covers : Graph.Covers PrimePairVertices.smallTree smallNode6173 :=
  Graph.node_covers _ 6173 smallRow6173 smallNode6163 smallEmpty
    smallNode6163_covers smallEmpty_covers (by decide) (by decide) smallRow6173_covers

def smallNode6199 : Graph := .node 6199 smallRow6199 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6199_covers : Graph.Covers PrimePairVertices.smallTree smallNode6199 :=
  Graph.node_covers _ 6199 smallRow6199 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6199_covers

def smallNode6203 : Graph := .node 6203 smallRow6203 smallNode6199 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6203_covers : Graph.Covers PrimePairVertices.smallTree smallNode6203 :=
  Graph.node_covers _ 6203 smallRow6203 smallNode6199 smallEmpty
    smallNode6199_covers smallEmpty_covers (by decide) (by decide) smallRow6203_covers

def smallNode6197 : Graph := .node 6197 smallRow6197 smallNode6173 smallNode6203
set_option maxRecDepth 4096 in
theorem smallNode6197_covers : Graph.Covers PrimePairVertices.smallTree smallNode6197 :=
  Graph.node_covers _ 6197 smallRow6197 smallNode6173 smallNode6203
    smallNode6173_covers smallNode6203_covers (by decide) (by decide) smallRow6197_covers

def smallNode6217 : Graph := .node 6217 smallRow6217 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6217_covers : Graph.Covers PrimePairVertices.smallTree smallNode6217 :=
  Graph.node_covers _ 6217 smallRow6217 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6217_covers

def smallNode6221 : Graph := .node 6221 smallRow6221 smallNode6217 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6221_covers : Graph.Covers PrimePairVertices.smallTree smallNode6221 :=
  Graph.node_covers _ 6221 smallRow6221 smallNode6217 smallEmpty
    smallNode6217_covers smallEmpty_covers (by decide) (by decide) smallRow6221_covers

def smallNode6247 : Graph := .node 6247 smallRow6247 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6247_covers : Graph.Covers PrimePairVertices.smallTree smallNode6247 :=
  Graph.node_covers _ 6247 smallRow6247 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6247_covers

def smallNode6229 : Graph := .node 6229 smallRow6229 smallNode6221 smallNode6247
set_option maxRecDepth 4096 in
theorem smallNode6229_covers : Graph.Covers PrimePairVertices.smallTree smallNode6229 :=
  Graph.node_covers _ 6229 smallRow6229 smallNode6221 smallNode6247
    smallNode6221_covers smallNode6247_covers (by decide) (by decide) smallRow6229_covers

def smallNode6211 : Graph := .node 6211 smallRow6211 smallNode6197 smallNode6229
set_option maxRecDepth 4096 in
theorem smallNode6211_covers : Graph.Covers PrimePairVertices.smallTree smallNode6211 :=
  Graph.node_covers _ 6211 smallRow6211 smallNode6197 smallNode6229
    smallNode6197_covers smallNode6229_covers (by decide) (by decide) smallRow6211_covers

def smallNode6263 : Graph := .node 6263 smallRow6263 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6263_covers : Graph.Covers PrimePairVertices.smallTree smallNode6263 :=
  Graph.node_covers _ 6263 smallRow6263 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6263_covers

def smallNode6269 : Graph := .node 6269 smallRow6269 smallNode6263 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6269_covers : Graph.Covers PrimePairVertices.smallTree smallNode6269 :=
  Graph.node_covers _ 6269 smallRow6269 smallNode6263 smallEmpty
    smallNode6263_covers smallEmpty_covers (by decide) (by decide) smallRow6269_covers

def smallNode6277 : Graph := .node 6277 smallRow6277 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6277_covers : Graph.Covers PrimePairVertices.smallTree smallNode6277 :=
  Graph.node_covers _ 6277 smallRow6277 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6277_covers

def smallNode6271 : Graph := .node 6271 smallRow6271 smallNode6269 smallNode6277
set_option maxRecDepth 4096 in
theorem smallNode6271_covers : Graph.Covers PrimePairVertices.smallTree smallNode6271 :=
  Graph.node_covers _ 6271 smallRow6271 smallNode6269 smallNode6277
    smallNode6269_covers smallNode6277_covers (by decide) (by decide) smallRow6271_covers

def smallNode6299 : Graph := .node 6299 smallRow6299 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6299_covers : Graph.Covers PrimePairVertices.smallTree smallNode6299 :=
  Graph.node_covers _ 6299 smallRow6299 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6299_covers

def smallNode6301 : Graph := .node 6301 smallRow6301 smallNode6299 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6301_covers : Graph.Covers PrimePairVertices.smallTree smallNode6301 :=
  Graph.node_covers _ 6301 smallRow6301 smallNode6299 smallEmpty
    smallNode6299_covers smallEmpty_covers (by decide) (by decide) smallRow6301_covers

def smallNode6317 : Graph := .node 6317 smallRow6317 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6317_covers : Graph.Covers PrimePairVertices.smallTree smallNode6317 :=
  Graph.node_covers _ 6317 smallRow6317 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6317_covers

def smallNode6311 : Graph := .node 6311 smallRow6311 smallNode6301 smallNode6317
set_option maxRecDepth 4096 in
theorem smallNode6311_covers : Graph.Covers PrimePairVertices.smallTree smallNode6311 :=
  Graph.node_covers _ 6311 smallRow6311 smallNode6301 smallNode6317
    smallNode6301_covers smallNode6317_covers (by decide) (by decide) smallRow6311_covers

def smallNode6287 : Graph := .node 6287 smallRow6287 smallNode6271 smallNode6311
set_option maxRecDepth 4096 in
theorem smallNode6287_covers : Graph.Covers PrimePairVertices.smallTree smallNode6287 :=
  Graph.node_covers _ 6287 smallRow6287 smallNode6271 smallNode6311
    smallNode6271_covers smallNode6311_covers (by decide) (by decide) smallRow6287_covers

def smallNode6257 : Graph := .node 6257 smallRow6257 smallNode6211 smallNode6287
set_option maxRecDepth 4096 in
theorem smallNode6257_covers : Graph.Covers PrimePairVertices.smallTree smallNode6257 :=
  Graph.node_covers _ 6257 smallRow6257 smallNode6211 smallNode6287
    smallNode6211_covers smallNode6287_covers (by decide) (by decide) smallRow6257_covers

def smallNode6329 : Graph := .node 6329 smallRow6329 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6329_covers : Graph.Covers PrimePairVertices.smallTree smallNode6329 :=
  Graph.node_covers _ 6329 smallRow6329 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6329_covers

def smallNode6337 : Graph := .node 6337 smallRow6337 smallNode6329 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6337_covers : Graph.Covers PrimePairVertices.smallTree smallNode6337 :=
  Graph.node_covers _ 6337 smallRow6337 smallNode6329 smallEmpty
    smallNode6329_covers smallEmpty_covers (by decide) (by decide) smallRow6337_covers

def smallNode6353 : Graph := .node 6353 smallRow6353 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6353_covers : Graph.Covers PrimePairVertices.smallTree smallNode6353 :=
  Graph.node_covers _ 6353 smallRow6353 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6353_covers

def smallNode6359 : Graph := .node 6359 smallRow6359 smallNode6353 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6359_covers : Graph.Covers PrimePairVertices.smallTree smallNode6359 :=
  Graph.node_covers _ 6359 smallRow6359 smallNode6353 smallEmpty
    smallNode6353_covers smallEmpty_covers (by decide) (by decide) smallRow6359_covers

def smallNode6343 : Graph := .node 6343 smallRow6343 smallNode6337 smallNode6359
set_option maxRecDepth 4096 in
theorem smallNode6343_covers : Graph.Covers PrimePairVertices.smallTree smallNode6343 :=
  Graph.node_covers _ 6343 smallRow6343 smallNode6337 smallNode6359
    smallNode6337_covers smallNode6359_covers (by decide) (by decide) smallRow6343_covers

def smallNode6367 : Graph := .node 6367 smallRow6367 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6367_covers : Graph.Covers PrimePairVertices.smallTree smallNode6367 :=
  Graph.node_covers _ 6367 smallRow6367 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6367_covers

def smallNode6373 : Graph := .node 6373 smallRow6373 smallNode6367 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6373_covers : Graph.Covers PrimePairVertices.smallTree smallNode6373 :=
  Graph.node_covers _ 6373 smallRow6373 smallNode6367 smallEmpty
    smallNode6367_covers smallEmpty_covers (by decide) (by decide) smallRow6373_covers

def smallNode6389 : Graph := .node 6389 smallRow6389 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6389_covers : Graph.Covers PrimePairVertices.smallTree smallNode6389 :=
  Graph.node_covers _ 6389 smallRow6389 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6389_covers

def smallNode6379 : Graph := .node 6379 smallRow6379 smallNode6373 smallNode6389
set_option maxRecDepth 4096 in
theorem smallNode6379_covers : Graph.Covers PrimePairVertices.smallTree smallNode6379 :=
  Graph.node_covers _ 6379 smallRow6379 smallNode6373 smallNode6389
    smallNode6373_covers smallNode6389_covers (by decide) (by decide) smallRow6379_covers

def smallNode6361 : Graph := .node 6361 smallRow6361 smallNode6343 smallNode6379
set_option maxRecDepth 4096 in
theorem smallNode6361_covers : Graph.Covers PrimePairVertices.smallTree smallNode6361 :=
  Graph.node_covers _ 6361 smallRow6361 smallNode6343 smallNode6379
    smallNode6343_covers smallNode6379_covers (by decide) (by decide) smallRow6361_covers

def smallNode6421 : Graph := .node 6421 smallRow6421 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6421_covers : Graph.Covers PrimePairVertices.smallTree smallNode6421 :=
  Graph.node_covers _ 6421 smallRow6421 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6421_covers

def smallNode6427 : Graph := .node 6427 smallRow6427 smallNode6421 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6427_covers : Graph.Covers PrimePairVertices.smallTree smallNode6427 :=
  Graph.node_covers _ 6427 smallRow6427 smallNode6421 smallEmpty
    smallNode6421_covers smallEmpty_covers (by decide) (by decide) smallRow6427_covers

def smallNode6451 : Graph := .node 6451 smallRow6451 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6451_covers : Graph.Covers PrimePairVertices.smallTree smallNode6451 :=
  Graph.node_covers _ 6451 smallRow6451 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6451_covers

def smallNode6449 : Graph := .node 6449 smallRow6449 smallNode6427 smallNode6451
set_option maxRecDepth 4096 in
theorem smallNode6449_covers : Graph.Covers PrimePairVertices.smallTree smallNode6449 :=
  Graph.node_covers _ 6449 smallRow6449 smallNode6427 smallNode6451
    smallNode6427_covers smallNode6451_covers (by decide) (by decide) smallRow6449_covers

def smallNode6473 : Graph := .node 6473 smallRow6473 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6473_covers : Graph.Covers PrimePairVertices.smallTree smallNode6473 :=
  Graph.node_covers _ 6473 smallRow6473 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6473_covers

def smallNode6481 : Graph := .node 6481 smallRow6481 smallNode6473 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6481_covers : Graph.Covers PrimePairVertices.smallTree smallNode6481 :=
  Graph.node_covers _ 6481 smallRow6481 smallNode6473 smallEmpty
    smallNode6473_covers smallEmpty_covers (by decide) (by decide) smallRow6481_covers

def smallNode6521 : Graph := .node 6521 smallRow6521 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6521_covers : Graph.Covers PrimePairVertices.smallTree smallNode6521 :=
  Graph.node_covers _ 6521 smallRow6521 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6521_covers

def smallNode6491 : Graph := .node 6491 smallRow6491 smallNode6481 smallNode6521
set_option maxRecDepth 4096 in
theorem smallNode6491_covers : Graph.Covers PrimePairVertices.smallTree smallNode6491 :=
  Graph.node_covers _ 6491 smallRow6491 smallNode6481 smallNode6521
    smallNode6481_covers smallNode6521_covers (by decide) (by decide) smallRow6491_covers

def smallNode6469 : Graph := .node 6469 smallRow6469 smallNode6449 smallNode6491
set_option maxRecDepth 4096 in
theorem smallNode6469_covers : Graph.Covers PrimePairVertices.smallTree smallNode6469 :=
  Graph.node_covers _ 6469 smallRow6469 smallNode6449 smallNode6491
    smallNode6449_covers smallNode6491_covers (by decide) (by decide) smallRow6469_covers

def smallNode6397 : Graph := .node 6397 smallRow6397 smallNode6361 smallNode6469
set_option maxRecDepth 4096 in
theorem smallNode6397_covers : Graph.Covers PrimePairVertices.smallTree smallNode6397 :=
  Graph.node_covers _ 6397 smallRow6397 smallNode6361 smallNode6469
    smallNode6361_covers smallNode6469_covers (by decide) (by decide) smallRow6397_covers

def smallNode6323 : Graph := .node 6323 smallRow6323 smallNode6257 smallNode6397
set_option maxRecDepth 4096 in
theorem smallNode6323_covers : Graph.Covers PrimePairVertices.smallTree smallNode6323 :=
  Graph.node_covers _ 6323 smallRow6323 smallNode6257 smallNode6397
    smallNode6257_covers smallNode6397_covers (by decide) (by decide) smallRow6323_covers

def smallNode6547 : Graph := .node 6547 smallRow6547 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6547_covers : Graph.Covers PrimePairVertices.smallTree smallNode6547 :=
  Graph.node_covers _ 6547 smallRow6547 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6547_covers

def smallNode6551 : Graph := .node 6551 smallRow6551 smallNode6547 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6551_covers : Graph.Covers PrimePairVertices.smallTree smallNode6551 :=
  Graph.node_covers _ 6551 smallRow6551 smallNode6547 smallEmpty
    smallNode6547_covers smallEmpty_covers (by decide) (by decide) smallRow6551_covers

def smallNode6563 : Graph := .node 6563 smallRow6563 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6563_covers : Graph.Covers PrimePairVertices.smallTree smallNode6563 :=
  Graph.node_covers _ 6563 smallRow6563 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6563_covers

def smallNode6569 : Graph := .node 6569 smallRow6569 smallNode6563 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6569_covers : Graph.Covers PrimePairVertices.smallTree smallNode6569 :=
  Graph.node_covers _ 6569 smallRow6569 smallNode6563 smallEmpty
    smallNode6563_covers smallEmpty_covers (by decide) (by decide) smallRow6569_covers

def smallNode6553 : Graph := .node 6553 smallRow6553 smallNode6551 smallNode6569
set_option maxRecDepth 4096 in
theorem smallNode6553_covers : Graph.Covers PrimePairVertices.smallTree smallNode6553 :=
  Graph.node_covers _ 6553 smallRow6553 smallNode6551 smallNode6569
    smallNode6551_covers smallNode6569_covers (by decide) (by decide) smallRow6553_covers

def smallNode6577 : Graph := .node 6577 smallRow6577 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6577_covers : Graph.Covers PrimePairVertices.smallTree smallNode6577 :=
  Graph.node_covers _ 6577 smallRow6577 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6577_covers

def smallNode6581 : Graph := .node 6581 smallRow6581 smallNode6577 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6581_covers : Graph.Covers PrimePairVertices.smallTree smallNode6581 :=
  Graph.node_covers _ 6581 smallRow6581 smallNode6577 smallEmpty
    smallNode6577_covers smallEmpty_covers (by decide) (by decide) smallRow6581_covers

def smallNode6607 : Graph := .node 6607 smallRow6607 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6607_covers : Graph.Covers PrimePairVertices.smallTree smallNode6607 :=
  Graph.node_covers _ 6607 smallRow6607 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6607_covers

def smallNode6599 : Graph := .node 6599 smallRow6599 smallNode6581 smallNode6607
set_option maxRecDepth 4096 in
theorem smallNode6599_covers : Graph.Covers PrimePairVertices.smallTree smallNode6599 :=
  Graph.node_covers _ 6599 smallRow6599 smallNode6581 smallNode6607
    smallNode6581_covers smallNode6607_covers (by decide) (by decide) smallRow6599_covers

def smallNode6571 : Graph := .node 6571 smallRow6571 smallNode6553 smallNode6599
set_option maxRecDepth 4096 in
theorem smallNode6571_covers : Graph.Covers PrimePairVertices.smallTree smallNode6571 :=
  Graph.node_covers _ 6571 smallRow6571 smallNode6553 smallNode6599
    smallNode6553_covers smallNode6599_covers (by decide) (by decide) smallRow6571_covers

def smallNode6637 : Graph := .node 6637 smallRow6637 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6637_covers : Graph.Covers PrimePairVertices.smallTree smallNode6637 :=
  Graph.node_covers _ 6637 smallRow6637 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6637_covers

def smallNode6653 : Graph := .node 6653 smallRow6653 smallNode6637 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6653_covers : Graph.Covers PrimePairVertices.smallTree smallNode6653 :=
  Graph.node_covers _ 6653 smallRow6653 smallNode6637 smallEmpty
    smallNode6637_covers smallEmpty_covers (by decide) (by decide) smallRow6653_covers

def smallNode6661 : Graph := .node 6661 smallRow6661 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6661_covers : Graph.Covers PrimePairVertices.smallTree smallNode6661 :=
  Graph.node_covers _ 6661 smallRow6661 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6661_covers

def smallNode6659 : Graph := .node 6659 smallRow6659 smallNode6653 smallNode6661
set_option maxRecDepth 4096 in
theorem smallNode6659_covers : Graph.Covers PrimePairVertices.smallTree smallNode6659 :=
  Graph.node_covers _ 6659 smallRow6659 smallNode6653 smallNode6661
    smallNode6653_covers smallNode6661_covers (by decide) (by decide) smallRow6659_covers

def smallNode6679 : Graph := .node 6679 smallRow6679 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6679_covers : Graph.Covers PrimePairVertices.smallTree smallNode6679 :=
  Graph.node_covers _ 6679 smallRow6679 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6679_covers

def smallNode6689 : Graph := .node 6689 smallRow6689 smallNode6679 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6689_covers : Graph.Covers PrimePairVertices.smallTree smallNode6689 :=
  Graph.node_covers _ 6689 smallRow6689 smallNode6679 smallEmpty
    smallNode6679_covers smallEmpty_covers (by decide) (by decide) smallRow6689_covers

def smallNode6701 : Graph := .node 6701 smallRow6701 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6701_covers : Graph.Covers PrimePairVertices.smallTree smallNode6701 :=
  Graph.node_covers _ 6701 smallRow6701 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6701_covers

def smallNode6691 : Graph := .node 6691 smallRow6691 smallNode6689 smallNode6701
set_option maxRecDepth 4096 in
theorem smallNode6691_covers : Graph.Covers PrimePairVertices.smallTree smallNode6691 :=
  Graph.node_covers _ 6691 smallRow6691 smallNode6689 smallNode6701
    smallNode6689_covers smallNode6701_covers (by decide) (by decide) smallRow6691_covers

def smallNode6673 : Graph := .node 6673 smallRow6673 smallNode6659 smallNode6691
set_option maxRecDepth 4096 in
theorem smallNode6673_covers : Graph.Covers PrimePairVertices.smallTree smallNode6673 :=
  Graph.node_covers _ 6673 smallRow6673 smallNode6659 smallNode6691
    smallNode6659_covers smallNode6691_covers (by decide) (by decide) smallRow6673_covers

def smallNode6619 : Graph := .node 6619 smallRow6619 smallNode6571 smallNode6673
set_option maxRecDepth 4096 in
theorem smallNode6619_covers : Graph.Covers PrimePairVertices.smallTree smallNode6619 :=
  Graph.node_covers _ 6619 smallRow6619 smallNode6571 smallNode6673
    smallNode6571_covers smallNode6673_covers (by decide) (by decide) smallRow6619_covers

def smallNode6709 : Graph := .node 6709 smallRow6709 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6709_covers : Graph.Covers PrimePairVertices.smallTree smallNode6709 :=
  Graph.node_covers _ 6709 smallRow6709 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6709_covers

def smallNode6719 : Graph := .node 6719 smallRow6719 smallNode6709 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6719_covers : Graph.Covers PrimePairVertices.smallTree smallNode6719 :=
  Graph.node_covers _ 6719 smallRow6719 smallNode6709 smallEmpty
    smallNode6709_covers smallEmpty_covers (by decide) (by decide) smallRow6719_covers

def smallNode6737 : Graph := .node 6737 smallRow6737 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6737_covers : Graph.Covers PrimePairVertices.smallTree smallNode6737 :=
  Graph.node_covers _ 6737 smallRow6737 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6737_covers

def smallNode6733 : Graph := .node 6733 smallRow6733 smallNode6719 smallNode6737
set_option maxRecDepth 4096 in
theorem smallNode6733_covers : Graph.Covers PrimePairVertices.smallTree smallNode6733 :=
  Graph.node_covers _ 6733 smallRow6733 smallNode6719 smallNode6737
    smallNode6719_covers smallNode6737_covers (by decide) (by decide) smallRow6733_covers

def smallNode6763 : Graph := .node 6763 smallRow6763 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6763_covers : Graph.Covers PrimePairVertices.smallTree smallNode6763 :=
  Graph.node_covers _ 6763 smallRow6763 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6763_covers

def smallNode6779 : Graph := .node 6779 smallRow6779 smallNode6763 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6779_covers : Graph.Covers PrimePairVertices.smallTree smallNode6779 :=
  Graph.node_covers _ 6779 smallRow6779 smallNode6763 smallEmpty
    smallNode6763_covers smallEmpty_covers (by decide) (by decide) smallRow6779_covers

def smallNode6791 : Graph := .node 6791 smallRow6791 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6791_covers : Graph.Covers PrimePairVertices.smallTree smallNode6791 :=
  Graph.node_covers _ 6791 smallRow6791 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6791_covers

def smallNode6781 : Graph := .node 6781 smallRow6781 smallNode6779 smallNode6791
set_option maxRecDepth 4096 in
theorem smallNode6781_covers : Graph.Covers PrimePairVertices.smallTree smallNode6781 :=
  Graph.node_covers _ 6781 smallRow6781 smallNode6779 smallNode6791
    smallNode6779_covers smallNode6791_covers (by decide) (by decide) smallRow6781_covers

def smallNode6761 : Graph := .node 6761 smallRow6761 smallNode6733 smallNode6781
set_option maxRecDepth 4096 in
theorem smallNode6761_covers : Graph.Covers PrimePairVertices.smallTree smallNode6761 :=
  Graph.node_covers _ 6761 smallRow6761 smallNode6733 smallNode6781
    smallNode6733_covers smallNode6781_covers (by decide) (by decide) smallRow6761_covers

def smallNode6803 : Graph := .node 6803 smallRow6803 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6803_covers : Graph.Covers PrimePairVertices.smallTree smallNode6803 :=
  Graph.node_covers _ 6803 smallRow6803 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6803_covers

def smallNode6823 : Graph := .node 6823 smallRow6823 smallNode6803 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6823_covers : Graph.Covers PrimePairVertices.smallTree smallNode6823 :=
  Graph.node_covers _ 6823 smallRow6823 smallNode6803 smallEmpty
    smallNode6803_covers smallEmpty_covers (by decide) (by decide) smallRow6823_covers

def smallNode6829 : Graph := .node 6829 smallRow6829 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6829_covers : Graph.Covers PrimePairVertices.smallTree smallNode6829 :=
  Graph.node_covers _ 6829 smallRow6829 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6829_covers

def smallNode6827 : Graph := .node 6827 smallRow6827 smallNode6823 smallNode6829
set_option maxRecDepth 4096 in
theorem smallNode6827_covers : Graph.Covers PrimePairVertices.smallTree smallNode6827 :=
  Graph.node_covers _ 6827 smallRow6827 smallNode6823 smallNode6829
    smallNode6823_covers smallNode6829_covers (by decide) (by decide) smallRow6827_covers

def smallNode6841 : Graph := .node 6841 smallRow6841 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6841_covers : Graph.Covers PrimePairVertices.smallTree smallNode6841 :=
  Graph.node_covers _ 6841 smallRow6841 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6841_covers

def smallNode6857 : Graph := .node 6857 smallRow6857 smallNode6841 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6857_covers : Graph.Covers PrimePairVertices.smallTree smallNode6857 :=
  Graph.node_covers _ 6857 smallRow6857 smallNode6841 smallEmpty
    smallNode6841_covers smallEmpty_covers (by decide) (by decide) smallRow6857_covers

def smallNode6869 : Graph := .node 6869 smallRow6869 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6869_covers : Graph.Covers PrimePairVertices.smallTree smallNode6869 :=
  Graph.node_covers _ 6869 smallRow6869 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6869_covers

def smallNode6863 : Graph := .node 6863 smallRow6863 smallNode6857 smallNode6869
set_option maxRecDepth 4096 in
theorem smallNode6863_covers : Graph.Covers PrimePairVertices.smallTree smallNode6863 :=
  Graph.node_covers _ 6863 smallRow6863 smallNode6857 smallNode6869
    smallNode6857_covers smallNode6869_covers (by decide) (by decide) smallRow6863_covers

def smallNode6833 : Graph := .node 6833 smallRow6833 smallNode6827 smallNode6863
set_option maxRecDepth 4096 in
theorem smallNode6833_covers : Graph.Covers PrimePairVertices.smallTree smallNode6833 :=
  Graph.node_covers _ 6833 smallRow6833 smallNode6827 smallNode6863
    smallNode6827_covers smallNode6863_covers (by decide) (by decide) smallRow6833_covers

def smallNode6793 : Graph := .node 6793 smallRow6793 smallNode6761 smallNode6833
set_option maxRecDepth 4096 in
theorem smallNode6793_covers : Graph.Covers PrimePairVertices.smallTree smallNode6793 :=
  Graph.node_covers _ 6793 smallRow6793 smallNode6761 smallNode6833
    smallNode6761_covers smallNode6833_covers (by decide) (by decide) smallRow6793_covers

def smallNode6703 : Graph := .node 6703 smallRow6703 smallNode6619 smallNode6793
set_option maxRecDepth 4096 in
theorem smallNode6703_covers : Graph.Covers PrimePairVertices.smallTree smallNode6703 :=
  Graph.node_covers _ 6703 smallRow6703 smallNode6619 smallNode6793
    smallNode6619_covers smallNode6793_covers (by decide) (by decide) smallRow6703_covers

def smallNode6529 : Graph := .node 6529 smallRow6529 smallNode6323 smallNode6703
set_option maxRecDepth 4096 in
theorem smallNode6529_covers : Graph.Covers PrimePairVertices.smallTree smallNode6529 :=
  Graph.node_covers _ 6529 smallRow6529 smallNode6323 smallNode6703
    smallNode6323_covers smallNode6703_covers (by decide) (by decide) smallRow6529_covers

def smallNode6151 : Graph := .node 6151 smallRow6151 smallNode5801 smallNode6529
set_option maxRecDepth 4096 in
theorem smallNode6151_covers : Graph.Covers PrimePairVertices.smallTree smallNode6151 :=
  Graph.node_covers _ 6151 smallRow6151 smallNode5801 smallNode6529
    smallNode5801_covers smallNode6529_covers (by decide) (by decide) smallRow6151_covers

def smallNode6883 : Graph := .node 6883 smallRow6883 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6883_covers : Graph.Covers PrimePairVertices.smallTree smallNode6883 :=
  Graph.node_covers _ 6883 smallRow6883 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6883_covers

def smallNode6899 : Graph := .node 6899 smallRow6899 smallNode6883 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6899_covers : Graph.Covers PrimePairVertices.smallTree smallNode6899 :=
  Graph.node_covers _ 6899 smallRow6899 smallNode6883 smallEmpty
    smallNode6883_covers smallEmpty_covers (by decide) (by decide) smallRow6899_covers

def smallNode6911 : Graph := .node 6911 smallRow6911 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6911_covers : Graph.Covers PrimePairVertices.smallTree smallNode6911 :=
  Graph.node_covers _ 6911 smallRow6911 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6911_covers

def smallNode6917 : Graph := .node 6917 smallRow6917 smallNode6911 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6917_covers : Graph.Covers PrimePairVertices.smallTree smallNode6917 :=
  Graph.node_covers _ 6917 smallRow6917 smallNode6911 smallEmpty
    smallNode6911_covers smallEmpty_covers (by decide) (by decide) smallRow6917_covers

def smallNode6907 : Graph := .node 6907 smallRow6907 smallNode6899 smallNode6917
set_option maxRecDepth 4096 in
theorem smallNode6907_covers : Graph.Covers PrimePairVertices.smallTree smallNode6907 :=
  Graph.node_covers _ 6907 smallRow6907 smallNode6899 smallNode6917
    smallNode6899_covers smallNode6917_covers (by decide) (by decide) smallRow6907_covers

def smallNode6949 : Graph := .node 6949 smallRow6949 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6949_covers : Graph.Covers PrimePairVertices.smallTree smallNode6949 :=
  Graph.node_covers _ 6949 smallRow6949 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6949_covers

def smallNode6959 : Graph := .node 6959 smallRow6959 smallNode6949 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6959_covers : Graph.Covers PrimePairVertices.smallTree smallNode6959 :=
  Graph.node_covers _ 6959 smallRow6959 smallNode6949 smallEmpty
    smallNode6949_covers smallEmpty_covers (by decide) (by decide) smallRow6959_covers

def smallNode6967 : Graph := .node 6967 smallRow6967 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6967_covers : Graph.Covers PrimePairVertices.smallTree smallNode6967 :=
  Graph.node_covers _ 6967 smallRow6967 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6967_covers

def smallNode6961 : Graph := .node 6961 smallRow6961 smallNode6959 smallNode6967
set_option maxRecDepth 4096 in
theorem smallNode6961_covers : Graph.Covers PrimePairVertices.smallTree smallNode6961 :=
  Graph.node_covers _ 6961 smallRow6961 smallNode6959 smallNode6967
    smallNode6959_covers smallNode6967_covers (by decide) (by decide) smallRow6961_covers

def smallNode6947 : Graph := .node 6947 smallRow6947 smallNode6907 smallNode6961
set_option maxRecDepth 4096 in
theorem smallNode6947_covers : Graph.Covers PrimePairVertices.smallTree smallNode6947 :=
  Graph.node_covers _ 6947 smallRow6947 smallNode6907 smallNode6961
    smallNode6907_covers smallNode6961_covers (by decide) (by decide) smallRow6947_covers

def smallNode6977 : Graph := .node 6977 smallRow6977 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6977_covers : Graph.Covers PrimePairVertices.smallTree smallNode6977 :=
  Graph.node_covers _ 6977 smallRow6977 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6977_covers

def smallNode6983 : Graph := .node 6983 smallRow6983 smallNode6977 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6983_covers : Graph.Covers PrimePairVertices.smallTree smallNode6983 :=
  Graph.node_covers _ 6983 smallRow6983 smallNode6977 smallEmpty
    smallNode6977_covers smallEmpty_covers (by decide) (by decide) smallRow6983_covers

def smallNode6997 : Graph := .node 6997 smallRow6997 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode6997_covers : Graph.Covers PrimePairVertices.smallTree smallNode6997 :=
  Graph.node_covers _ 6997 smallRow6997 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow6997_covers

def smallNode6991 : Graph := .node 6991 smallRow6991 smallNode6983 smallNode6997
set_option maxRecDepth 4096 in
theorem smallNode6991_covers : Graph.Covers PrimePairVertices.smallTree smallNode6991 :=
  Graph.node_covers _ 6991 smallRow6991 smallNode6983 smallNode6997
    smallNode6983_covers smallNode6997_covers (by decide) (by decide) smallRow6991_covers

def smallNode7013 : Graph := .node 7013 smallRow7013 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7013_covers : Graph.Covers PrimePairVertices.smallTree smallNode7013 :=
  Graph.node_covers _ 7013 smallRow7013 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7013_covers

def smallNode7019 : Graph := .node 7019 smallRow7019 smallNode7013 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7019_covers : Graph.Covers PrimePairVertices.smallTree smallNode7019 :=
  Graph.node_covers _ 7019 smallRow7019 smallNode7013 smallEmpty
    smallNode7013_covers smallEmpty_covers (by decide) (by decide) smallRow7019_covers

def smallNode7039 : Graph := .node 7039 smallRow7039 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7039_covers : Graph.Covers PrimePairVertices.smallTree smallNode7039 :=
  Graph.node_covers _ 7039 smallRow7039 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7039_covers

def smallNode7027 : Graph := .node 7027 smallRow7027 smallNode7019 smallNode7039
set_option maxRecDepth 4096 in
theorem smallNode7027_covers : Graph.Covers PrimePairVertices.smallTree smallNode7027 :=
  Graph.node_covers _ 7027 smallRow7027 smallNode7019 smallNode7039
    smallNode7019_covers smallNode7039_covers (by decide) (by decide) smallRow7027_covers

def smallNode7001 : Graph := .node 7001 smallRow7001 smallNode6991 smallNode7027
set_option maxRecDepth 4096 in
theorem smallNode7001_covers : Graph.Covers PrimePairVertices.smallTree smallNode7001 :=
  Graph.node_covers _ 7001 smallRow7001 smallNode6991 smallNode7027
    smallNode6991_covers smallNode7027_covers (by decide) (by decide) smallRow7001_covers

def smallNode6971 : Graph := .node 6971 smallRow6971 smallNode6947 smallNode7001
set_option maxRecDepth 4096 in
theorem smallNode6971_covers : Graph.Covers PrimePairVertices.smallTree smallNode6971 :=
  Graph.node_covers _ 6971 smallRow6971 smallNode6947 smallNode7001
    smallNode6947_covers smallNode7001_covers (by decide) (by decide) smallRow6971_covers

def smallNode7057 : Graph := .node 7057 smallRow7057 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7057_covers : Graph.Covers PrimePairVertices.smallTree smallNode7057 :=
  Graph.node_covers _ 7057 smallRow7057 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7057_covers

def smallNode7069 : Graph := .node 7069 smallRow7069 smallNode7057 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7069_covers : Graph.Covers PrimePairVertices.smallTree smallNode7069 :=
  Graph.node_covers _ 7069 smallRow7069 smallNode7057 smallEmpty
    smallNode7057_covers smallEmpty_covers (by decide) (by decide) smallRow7069_covers

def smallNode7103 : Graph := .node 7103 smallRow7103 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7103_covers : Graph.Covers PrimePairVertices.smallTree smallNode7103 :=
  Graph.node_covers _ 7103 smallRow7103 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7103_covers

def smallNode7109 : Graph := .node 7109 smallRow7109 smallNode7103 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7109_covers : Graph.Covers PrimePairVertices.smallTree smallNode7109 :=
  Graph.node_covers _ 7109 smallRow7109 smallNode7103 smallEmpty
    smallNode7103_covers smallEmpty_covers (by decide) (by decide) smallRow7109_covers

def smallNode7079 : Graph := .node 7079 smallRow7079 smallNode7069 smallNode7109
set_option maxRecDepth 4096 in
theorem smallNode7079_covers : Graph.Covers PrimePairVertices.smallTree smallNode7079 :=
  Graph.node_covers _ 7079 smallRow7079 smallNode7069 smallNode7109
    smallNode7069_covers smallNode7109_covers (by decide) (by decide) smallRow7079_covers

def smallNode7127 : Graph := .node 7127 smallRow7127 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7127_covers : Graph.Covers PrimePairVertices.smallTree smallNode7127 :=
  Graph.node_covers _ 7127 smallRow7127 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7127_covers

def smallNode7129 : Graph := .node 7129 smallRow7129 smallNode7127 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7129_covers : Graph.Covers PrimePairVertices.smallTree smallNode7129 :=
  Graph.node_covers _ 7129 smallRow7129 smallNode7127 smallEmpty
    smallNode7127_covers smallEmpty_covers (by decide) (by decide) smallRow7129_covers

def smallNode7159 : Graph := .node 7159 smallRow7159 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7159_covers : Graph.Covers PrimePairVertices.smallTree smallNode7159 :=
  Graph.node_covers _ 7159 smallRow7159 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7159_covers

def smallNode7151 : Graph := .node 7151 smallRow7151 smallNode7129 smallNode7159
set_option maxRecDepth 4096 in
theorem smallNode7151_covers : Graph.Covers PrimePairVertices.smallTree smallNode7151 :=
  Graph.node_covers _ 7151 smallRow7151 smallNode7129 smallNode7159
    smallNode7129_covers smallNode7159_covers (by decide) (by decide) smallRow7151_covers

def smallNode7121 : Graph := .node 7121 smallRow7121 smallNode7079 smallNode7151
set_option maxRecDepth 4096 in
theorem smallNode7121_covers : Graph.Covers PrimePairVertices.smallTree smallNode7121 :=
  Graph.node_covers _ 7121 smallRow7121 smallNode7079 smallNode7151
    smallNode7079_covers smallNode7151_covers (by decide) (by decide) smallRow7121_covers

def smallNode7187 : Graph := .node 7187 smallRow7187 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7187_covers : Graph.Covers PrimePairVertices.smallTree smallNode7187 :=
  Graph.node_covers _ 7187 smallRow7187 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7187_covers

def smallNode7193 : Graph := .node 7193 smallRow7193 smallNode7187 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7193_covers : Graph.Covers PrimePairVertices.smallTree smallNode7193 :=
  Graph.node_covers _ 7193 smallRow7193 smallNode7187 smallEmpty
    smallNode7187_covers smallEmpty_covers (by decide) (by decide) smallRow7193_covers

def smallNode7211 : Graph := .node 7211 smallRow7211 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7211_covers : Graph.Covers PrimePairVertices.smallTree smallNode7211 :=
  Graph.node_covers _ 7211 smallRow7211 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7211_covers

def smallNode7207 : Graph := .node 7207 smallRow7207 smallNode7193 smallNode7211
set_option maxRecDepth 4096 in
theorem smallNode7207_covers : Graph.Covers PrimePairVertices.smallTree smallNode7207 :=
  Graph.node_covers _ 7207 smallRow7207 smallNode7193 smallNode7211
    smallNode7193_covers smallNode7211_covers (by decide) (by decide) smallRow7207_covers

def smallNode7219 : Graph := .node 7219 smallRow7219 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7219_covers : Graph.Covers PrimePairVertices.smallTree smallNode7219 :=
  Graph.node_covers _ 7219 smallRow7219 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7219_covers

def smallNode7229 : Graph := .node 7229 smallRow7229 smallNode7219 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7229_covers : Graph.Covers PrimePairVertices.smallTree smallNode7229 :=
  Graph.node_covers _ 7229 smallRow7229 smallNode7219 smallEmpty
    smallNode7219_covers smallEmpty_covers (by decide) (by decide) smallRow7229_covers

def smallNode7243 : Graph := .node 7243 smallRow7243 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7243_covers : Graph.Covers PrimePairVertices.smallTree smallNode7243 :=
  Graph.node_covers _ 7243 smallRow7243 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7243_covers

def smallNode7237 : Graph := .node 7237 smallRow7237 smallNode7229 smallNode7243
set_option maxRecDepth 4096 in
theorem smallNode7237_covers : Graph.Covers PrimePairVertices.smallTree smallNode7237 :=
  Graph.node_covers _ 7237 smallRow7237 smallNode7229 smallNode7243
    smallNode7229_covers smallNode7243_covers (by decide) (by decide) smallRow7237_covers

def smallNode7213 : Graph := .node 7213 smallRow7213 smallNode7207 smallNode7237
set_option maxRecDepth 4096 in
theorem smallNode7213_covers : Graph.Covers PrimePairVertices.smallTree smallNode7213 :=
  Graph.node_covers _ 7213 smallRow7213 smallNode7207 smallNode7237
    smallNode7207_covers smallNode7237_covers (by decide) (by decide) smallRow7213_covers

def smallNode7177 : Graph := .node 7177 smallRow7177 smallNode7121 smallNode7213
set_option maxRecDepth 4096 in
theorem smallNode7177_covers : Graph.Covers PrimePairVertices.smallTree smallNode7177 :=
  Graph.node_covers _ 7177 smallRow7177 smallNode7121 smallNode7213
    smallNode7121_covers smallNode7213_covers (by decide) (by decide) smallRow7177_covers

def smallNode7043 : Graph := .node 7043 smallRow7043 smallNode6971 smallNode7177
set_option maxRecDepth 4096 in
theorem smallNode7043_covers : Graph.Covers PrimePairVertices.smallTree smallNode7043 :=
  Graph.node_covers _ 7043 smallRow7043 smallNode6971 smallNode7177
    smallNode6971_covers smallNode7177_covers (by decide) (by decide) smallRow7043_covers

def smallNode7253 : Graph := .node 7253 smallRow7253 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7253_covers : Graph.Covers PrimePairVertices.smallTree smallNode7253 :=
  Graph.node_covers _ 7253 smallRow7253 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7253_covers

def smallNode7283 : Graph := .node 7283 smallRow7283 smallNode7253 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7283_covers : Graph.Covers PrimePairVertices.smallTree smallNode7283 :=
  Graph.node_covers _ 7283 smallRow7283 smallNode7253 smallEmpty
    smallNode7253_covers smallEmpty_covers (by decide) (by decide) smallRow7283_covers

def smallNode7307 : Graph := .node 7307 smallRow7307 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7307_covers : Graph.Covers PrimePairVertices.smallTree smallNode7307 :=
  Graph.node_covers _ 7307 smallRow7307 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7307_covers

def smallNode7309 : Graph := .node 7309 smallRow7309 smallNode7307 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7309_covers : Graph.Covers PrimePairVertices.smallTree smallNode7309 :=
  Graph.node_covers _ 7309 smallRow7309 smallNode7307 smallEmpty
    smallNode7307_covers smallEmpty_covers (by decide) (by decide) smallRow7309_covers

def smallNode7297 : Graph := .node 7297 smallRow7297 smallNode7283 smallNode7309
set_option maxRecDepth 4096 in
theorem smallNode7297_covers : Graph.Covers PrimePairVertices.smallTree smallNode7297 :=
  Graph.node_covers _ 7297 smallRow7297 smallNode7283 smallNode7309
    smallNode7283_covers smallNode7309_covers (by decide) (by decide) smallRow7297_covers

def smallNode7331 : Graph := .node 7331 smallRow7331 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7331_covers : Graph.Covers PrimePairVertices.smallTree smallNode7331 :=
  Graph.node_covers _ 7331 smallRow7331 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7331_covers

def smallNode7333 : Graph := .node 7333 smallRow7333 smallNode7331 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7333_covers : Graph.Covers PrimePairVertices.smallTree smallNode7333 :=
  Graph.node_covers _ 7333 smallRow7333 smallNode7331 smallEmpty
    smallNode7331_covers smallEmpty_covers (by decide) (by decide) smallRow7333_covers

def smallNode7351 : Graph := .node 7351 smallRow7351 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7351_covers : Graph.Covers PrimePairVertices.smallTree smallNode7351 :=
  Graph.node_covers _ 7351 smallRow7351 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7351_covers

def smallNode7349 : Graph := .node 7349 smallRow7349 smallNode7333 smallNode7351
set_option maxRecDepth 4096 in
theorem smallNode7349_covers : Graph.Covers PrimePairVertices.smallTree smallNode7349 :=
  Graph.node_covers _ 7349 smallRow7349 smallNode7333 smallNode7351
    smallNode7333_covers smallNode7351_covers (by decide) (by decide) smallRow7349_covers

def smallNode7321 : Graph := .node 7321 smallRow7321 smallNode7297 smallNode7349
set_option maxRecDepth 4096 in
theorem smallNode7321_covers : Graph.Covers PrimePairVertices.smallTree smallNode7321 :=
  Graph.node_covers _ 7321 smallRow7321 smallNode7297 smallNode7349
    smallNode7297_covers smallNode7349_covers (by decide) (by decide) smallRow7321_covers

def smallNode7393 : Graph := .node 7393 smallRow7393 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7393_covers : Graph.Covers PrimePairVertices.smallTree smallNode7393 :=
  Graph.node_covers _ 7393 smallRow7393 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7393_covers

def smallNode7411 : Graph := .node 7411 smallRow7411 smallNode7393 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7411_covers : Graph.Covers PrimePairVertices.smallTree smallNode7411 :=
  Graph.node_covers _ 7411 smallRow7411 smallNode7393 smallEmpty
    smallNode7393_covers smallEmpty_covers (by decide) (by decide) smallRow7411_covers

def smallNode7433 : Graph := .node 7433 smallRow7433 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7433_covers : Graph.Covers PrimePairVertices.smallTree smallNode7433 :=
  Graph.node_covers _ 7433 smallRow7433 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7433_covers

def smallNode7417 : Graph := .node 7417 smallRow7417 smallNode7411 smallNode7433
set_option maxRecDepth 4096 in
theorem smallNode7417_covers : Graph.Covers PrimePairVertices.smallTree smallNode7417 :=
  Graph.node_covers _ 7417 smallRow7417 smallNode7411 smallNode7433
    smallNode7411_covers smallNode7433_covers (by decide) (by decide) smallRow7417_covers

def smallNode7457 : Graph := .node 7457 smallRow7457 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7457_covers : Graph.Covers PrimePairVertices.smallTree smallNode7457 :=
  Graph.node_covers _ 7457 smallRow7457 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7457_covers

def smallNode7459 : Graph := .node 7459 smallRow7459 smallNode7457 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7459_covers : Graph.Covers PrimePairVertices.smallTree smallNode7459 :=
  Graph.node_covers _ 7459 smallRow7459 smallNode7457 smallEmpty
    smallNode7457_covers smallEmpty_covers (by decide) (by decide) smallRow7459_covers

def smallNode7481 : Graph := .node 7481 smallRow7481 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7481_covers : Graph.Covers PrimePairVertices.smallTree smallNode7481 :=
  Graph.node_covers _ 7481 smallRow7481 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7481_covers

def smallNode7477 : Graph := .node 7477 smallRow7477 smallNode7459 smallNode7481
set_option maxRecDepth 4096 in
theorem smallNode7477_covers : Graph.Covers PrimePairVertices.smallTree smallNode7477 :=
  Graph.node_covers _ 7477 smallRow7477 smallNode7459 smallNode7481
    smallNode7459_covers smallNode7481_covers (by decide) (by decide) smallRow7477_covers

def smallNode7451 : Graph := .node 7451 smallRow7451 smallNode7417 smallNode7477
set_option maxRecDepth 4096 in
theorem smallNode7451_covers : Graph.Covers PrimePairVertices.smallTree smallNode7451 :=
  Graph.node_covers _ 7451 smallRow7451 smallNode7417 smallNode7477
    smallNode7417_covers smallNode7477_covers (by decide) (by decide) smallRow7451_covers

def smallNode7369 : Graph := .node 7369 smallRow7369 smallNode7321 smallNode7451
set_option maxRecDepth 4096 in
theorem smallNode7369_covers : Graph.Covers PrimePairVertices.smallTree smallNode7369 :=
  Graph.node_covers _ 7369 smallRow7369 smallNode7321 smallNode7451
    smallNode7321_covers smallNode7451_covers (by decide) (by decide) smallRow7369_covers

def smallNode7489 : Graph := .node 7489 smallRow7489 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7489_covers : Graph.Covers PrimePairVertices.smallTree smallNode7489 :=
  Graph.node_covers _ 7489 smallRow7489 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7489_covers

def smallNode7499 : Graph := .node 7499 smallRow7499 smallNode7489 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7499_covers : Graph.Covers PrimePairVertices.smallTree smallNode7499 :=
  Graph.node_covers _ 7499 smallRow7499 smallNode7489 smallEmpty
    smallNode7489_covers smallEmpty_covers (by decide) (by decide) smallRow7499_covers

def smallNode7517 : Graph := .node 7517 smallRow7517 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7517_covers : Graph.Covers PrimePairVertices.smallTree smallNode7517 :=
  Graph.node_covers _ 7517 smallRow7517 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7517_covers

def smallNode7507 : Graph := .node 7507 smallRow7507 smallNode7499 smallNode7517
set_option maxRecDepth 4096 in
theorem smallNode7507_covers : Graph.Covers PrimePairVertices.smallTree smallNode7507 :=
  Graph.node_covers _ 7507 smallRow7507 smallNode7499 smallNode7517
    smallNode7499_covers smallNode7517_covers (by decide) (by decide) smallRow7507_covers

def smallNode7529 : Graph := .node 7529 smallRow7529 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7529_covers : Graph.Covers PrimePairVertices.smallTree smallNode7529 :=
  Graph.node_covers _ 7529 smallRow7529 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7529_covers

def smallNode7537 : Graph := .node 7537 smallRow7537 smallNode7529 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7537_covers : Graph.Covers PrimePairVertices.smallTree smallNode7537 :=
  Graph.node_covers _ 7537 smallRow7537 smallNode7529 smallEmpty
    smallNode7529_covers smallEmpty_covers (by decide) (by decide) smallRow7537_covers

def smallNode7547 : Graph := .node 7547 smallRow7547 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7547_covers : Graph.Covers PrimePairVertices.smallTree smallNode7547 :=
  Graph.node_covers _ 7547 smallRow7547 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7547_covers

def smallNode7541 : Graph := .node 7541 smallRow7541 smallNode7537 smallNode7547
set_option maxRecDepth 4096 in
theorem smallNode7541_covers : Graph.Covers PrimePairVertices.smallTree smallNode7541 :=
  Graph.node_covers _ 7541 smallRow7541 smallNode7537 smallNode7547
    smallNode7537_covers smallNode7547_covers (by decide) (by decide) smallRow7541_covers

def smallNode7523 : Graph := .node 7523 smallRow7523 smallNode7507 smallNode7541
set_option maxRecDepth 4096 in
theorem smallNode7523_covers : Graph.Covers PrimePairVertices.smallTree smallNode7523 :=
  Graph.node_covers _ 7523 smallRow7523 smallNode7507 smallNode7541
    smallNode7507_covers smallNode7541_covers (by decide) (by decide) smallRow7523_covers

def smallNode7559 : Graph := .node 7559 smallRow7559 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7559_covers : Graph.Covers PrimePairVertices.smallTree smallNode7559 :=
  Graph.node_covers _ 7559 smallRow7559 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7559_covers

def smallNode7561 : Graph := .node 7561 smallRow7561 smallNode7559 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7561_covers : Graph.Covers PrimePairVertices.smallTree smallNode7561 :=
  Graph.node_covers _ 7561 smallRow7561 smallNode7559 smallEmpty
    smallNode7559_covers smallEmpty_covers (by decide) (by decide) smallRow7561_covers

def smallNode7577 : Graph := .node 7577 smallRow7577 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7577_covers : Graph.Covers PrimePairVertices.smallTree smallNode7577 :=
  Graph.node_covers _ 7577 smallRow7577 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7577_covers

def smallNode7573 : Graph := .node 7573 smallRow7573 smallNode7561 smallNode7577
set_option maxRecDepth 4096 in
theorem smallNode7573_covers : Graph.Covers PrimePairVertices.smallTree smallNode7573 :=
  Graph.node_covers _ 7573 smallRow7573 smallNode7561 smallNode7577
    smallNode7561_covers smallNode7577_covers (by decide) (by decide) smallRow7573_covers

def smallNode7589 : Graph := .node 7589 smallRow7589 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7589_covers : Graph.Covers PrimePairVertices.smallTree smallNode7589 :=
  Graph.node_covers _ 7589 smallRow7589 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7589_covers

def smallNode7591 : Graph := .node 7591 smallRow7591 smallNode7589 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7591_covers : Graph.Covers PrimePairVertices.smallTree smallNode7591 :=
  Graph.node_covers _ 7591 smallRow7591 smallNode7589 smallEmpty
    smallNode7589_covers smallEmpty_covers (by decide) (by decide) smallRow7591_covers

def smallNode7607 : Graph := .node 7607 smallRow7607 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7607_covers : Graph.Covers PrimePairVertices.smallTree smallNode7607 :=
  Graph.node_covers _ 7607 smallRow7607 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7607_covers

def smallNode7603 : Graph := .node 7603 smallRow7603 smallNode7591 smallNode7607
set_option maxRecDepth 4096 in
theorem smallNode7603_covers : Graph.Covers PrimePairVertices.smallTree smallNode7603 :=
  Graph.node_covers _ 7603 smallRow7603 smallNode7591 smallNode7607
    smallNode7591_covers smallNode7607_covers (by decide) (by decide) smallRow7603_covers

def smallNode7583 : Graph := .node 7583 smallRow7583 smallNode7573 smallNode7603
set_option maxRecDepth 4096 in
theorem smallNode7583_covers : Graph.Covers PrimePairVertices.smallTree smallNode7583 :=
  Graph.node_covers _ 7583 smallRow7583 smallNode7573 smallNode7603
    smallNode7573_covers smallNode7603_covers (by decide) (by decide) smallRow7583_covers

def smallNode7549 : Graph := .node 7549 smallRow7549 smallNode7523 smallNode7583
set_option maxRecDepth 4096 in
theorem smallNode7549_covers : Graph.Covers PrimePairVertices.smallTree smallNode7549 :=
  Graph.node_covers _ 7549 smallRow7549 smallNode7523 smallNode7583
    smallNode7523_covers smallNode7583_covers (by decide) (by decide) smallRow7549_covers

def smallNode7487 : Graph := .node 7487 smallRow7487 smallNode7369 smallNode7549
set_option maxRecDepth 4096 in
theorem smallNode7487_covers : Graph.Covers PrimePairVertices.smallTree smallNode7487 :=
  Graph.node_covers _ 7487 smallRow7487 smallNode7369 smallNode7549
    smallNode7369_covers smallNode7549_covers (by decide) (by decide) smallRow7487_covers

def smallNode7247 : Graph := .node 7247 smallRow7247 smallNode7043 smallNode7487
set_option maxRecDepth 4096 in
theorem smallNode7247_covers : Graph.Covers PrimePairVertices.smallTree smallNode7247 :=
  Graph.node_covers _ 7247 smallRow7247 smallNode7043 smallNode7487
    smallNode7043_covers smallNode7487_covers (by decide) (by decide) smallRow7247_covers

def smallNode7639 : Graph := .node 7639 smallRow7639 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7639_covers : Graph.Covers PrimePairVertices.smallTree smallNode7639 :=
  Graph.node_covers _ 7639 smallRow7639 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7639_covers

def smallNode7643 : Graph := .node 7643 smallRow7643 smallNode7639 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7643_covers : Graph.Covers PrimePairVertices.smallTree smallNode7643 :=
  Graph.node_covers _ 7643 smallRow7643 smallNode7639 smallEmpty
    smallNode7639_covers smallEmpty_covers (by decide) (by decide) smallRow7643_covers

def smallNode7669 : Graph := .node 7669 smallRow7669 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7669_covers : Graph.Covers PrimePairVertices.smallTree smallNode7669 :=
  Graph.node_covers _ 7669 smallRow7669 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7669_covers

def smallNode7673 : Graph := .node 7673 smallRow7673 smallNode7669 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7673_covers : Graph.Covers PrimePairVertices.smallTree smallNode7673 :=
  Graph.node_covers _ 7673 smallRow7673 smallNode7669 smallEmpty
    smallNode7669_covers smallEmpty_covers (by decide) (by decide) smallRow7673_covers

def smallNode7649 : Graph := .node 7649 smallRow7649 smallNode7643 smallNode7673
set_option maxRecDepth 4096 in
theorem smallNode7649_covers : Graph.Covers PrimePairVertices.smallTree smallNode7649 :=
  Graph.node_covers _ 7649 smallRow7649 smallNode7643 smallNode7673
    smallNode7643_covers smallNode7673_covers (by decide) (by decide) smallRow7649_covers

def smallNode7687 : Graph := .node 7687 smallRow7687 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7687_covers : Graph.Covers PrimePairVertices.smallTree smallNode7687 :=
  Graph.node_covers _ 7687 smallRow7687 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7687_covers

def smallNode7691 : Graph := .node 7691 smallRow7691 smallNode7687 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7691_covers : Graph.Covers PrimePairVertices.smallTree smallNode7691 :=
  Graph.node_covers _ 7691 smallRow7691 smallNode7687 smallEmpty
    smallNode7687_covers smallEmpty_covers (by decide) (by decide) smallRow7691_covers

def smallNode7703 : Graph := .node 7703 smallRow7703 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7703_covers : Graph.Covers PrimePairVertices.smallTree smallNode7703 :=
  Graph.node_covers _ 7703 smallRow7703 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7703_covers

def smallNode7699 : Graph := .node 7699 smallRow7699 smallNode7691 smallNode7703
set_option maxRecDepth 4096 in
theorem smallNode7699_covers : Graph.Covers PrimePairVertices.smallTree smallNode7699 :=
  Graph.node_covers _ 7699 smallRow7699 smallNode7691 smallNode7703
    smallNode7691_covers smallNode7703_covers (by decide) (by decide) smallRow7699_covers

def smallNode7681 : Graph := .node 7681 smallRow7681 smallNode7649 smallNode7699
set_option maxRecDepth 4096 in
theorem smallNode7681_covers : Graph.Covers PrimePairVertices.smallTree smallNode7681 :=
  Graph.node_covers _ 7681 smallRow7681 smallNode7649 smallNode7699
    smallNode7649_covers smallNode7699_covers (by decide) (by decide) smallRow7681_covers

def smallNode7723 : Graph := .node 7723 smallRow7723 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7723_covers : Graph.Covers PrimePairVertices.smallTree smallNode7723 :=
  Graph.node_covers _ 7723 smallRow7723 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7723_covers

def smallNode7727 : Graph := .node 7727 smallRow7727 smallNode7723 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7727_covers : Graph.Covers PrimePairVertices.smallTree smallNode7727 :=
  Graph.node_covers _ 7727 smallRow7727 smallNode7723 smallEmpty
    smallNode7723_covers smallEmpty_covers (by decide) (by decide) smallRow7727_covers

def smallNode7753 : Graph := .node 7753 smallRow7753 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7753_covers : Graph.Covers PrimePairVertices.smallTree smallNode7753 :=
  Graph.node_covers _ 7753 smallRow7753 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7753_covers

def smallNode7741 : Graph := .node 7741 smallRow7741 smallNode7727 smallNode7753
set_option maxRecDepth 4096 in
theorem smallNode7741_covers : Graph.Covers PrimePairVertices.smallTree smallNode7741 :=
  Graph.node_covers _ 7741 smallRow7741 smallNode7727 smallNode7753
    smallNode7727_covers smallNode7753_covers (by decide) (by decide) smallRow7741_covers

def smallNode7759 : Graph := .node 7759 smallRow7759 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7759_covers : Graph.Covers PrimePairVertices.smallTree smallNode7759 :=
  Graph.node_covers _ 7759 smallRow7759 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7759_covers

def smallNode7789 : Graph := .node 7789 smallRow7789 smallNode7759 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7789_covers : Graph.Covers PrimePairVertices.smallTree smallNode7789 :=
  Graph.node_covers _ 7789 smallRow7789 smallNode7759 smallEmpty
    smallNode7759_covers smallEmpty_covers (by decide) (by decide) smallRow7789_covers

def smallNode7817 : Graph := .node 7817 smallRow7817 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7817_covers : Graph.Covers PrimePairVertices.smallTree smallNode7817 :=
  Graph.node_covers _ 7817 smallRow7817 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7817_covers

def smallNode7793 : Graph := .node 7793 smallRow7793 smallNode7789 smallNode7817
set_option maxRecDepth 4096 in
theorem smallNode7793_covers : Graph.Covers PrimePairVertices.smallTree smallNode7793 :=
  Graph.node_covers _ 7793 smallRow7793 smallNode7789 smallNode7817
    smallNode7789_covers smallNode7817_covers (by decide) (by decide) smallRow7793_covers

def smallNode7757 : Graph := .node 7757 smallRow7757 smallNode7741 smallNode7793
set_option maxRecDepth 4096 in
theorem smallNode7757_covers : Graph.Covers PrimePairVertices.smallTree smallNode7757 :=
  Graph.node_covers _ 7757 smallRow7757 smallNode7741 smallNode7793
    smallNode7741_covers smallNode7793_covers (by decide) (by decide) smallRow7757_covers

def smallNode7717 : Graph := .node 7717 smallRow7717 smallNode7681 smallNode7757
set_option maxRecDepth 4096 in
theorem smallNode7717_covers : Graph.Covers PrimePairVertices.smallTree smallNode7717 :=
  Graph.node_covers _ 7717 smallRow7717 smallNode7681 smallNode7757
    smallNode7681_covers smallNode7757_covers (by decide) (by decide) smallRow7717_covers

def smallNode7829 : Graph := .node 7829 smallRow7829 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7829_covers : Graph.Covers PrimePairVertices.smallTree smallNode7829 :=
  Graph.node_covers _ 7829 smallRow7829 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7829_covers

def smallNode7841 : Graph := .node 7841 smallRow7841 smallNode7829 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7841_covers : Graph.Covers PrimePairVertices.smallTree smallNode7841 :=
  Graph.node_covers _ 7841 smallRow7841 smallNode7829 smallEmpty
    smallNode7829_covers smallEmpty_covers (by decide) (by decide) smallRow7841_covers

def smallNode7867 : Graph := .node 7867 smallRow7867 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7867_covers : Graph.Covers PrimePairVertices.smallTree smallNode7867 :=
  Graph.node_covers _ 7867 smallRow7867 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7867_covers

def smallNode7873 : Graph := .node 7873 smallRow7873 smallNode7867 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7873_covers : Graph.Covers PrimePairVertices.smallTree smallNode7873 :=
  Graph.node_covers _ 7873 smallRow7873 smallNode7867 smallEmpty
    smallNode7867_covers smallEmpty_covers (by decide) (by decide) smallRow7873_covers

def smallNode7853 : Graph := .node 7853 smallRow7853 smallNode7841 smallNode7873
set_option maxRecDepth 4096 in
theorem smallNode7853_covers : Graph.Covers PrimePairVertices.smallTree smallNode7853 :=
  Graph.node_covers _ 7853 smallRow7853 smallNode7841 smallNode7873
    smallNode7841_covers smallNode7873_covers (by decide) (by decide) smallRow7853_covers

def smallNode7879 : Graph := .node 7879 smallRow7879 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7879_covers : Graph.Covers PrimePairVertices.smallTree smallNode7879 :=
  Graph.node_covers _ 7879 smallRow7879 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7879_covers

def smallNode7883 : Graph := .node 7883 smallRow7883 smallNode7879 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7883_covers : Graph.Covers PrimePairVertices.smallTree smallNode7883 :=
  Graph.node_covers _ 7883 smallRow7883 smallNode7879 smallEmpty
    smallNode7879_covers smallEmpty_covers (by decide) (by decide) smallRow7883_covers

def smallNode7907 : Graph := .node 7907 smallRow7907 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7907_covers : Graph.Covers PrimePairVertices.smallTree smallNode7907 :=
  Graph.node_covers _ 7907 smallRow7907 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7907_covers

def smallNode7901 : Graph := .node 7901 smallRow7901 smallNode7883 smallNode7907
set_option maxRecDepth 4096 in
theorem smallNode7901_covers : Graph.Covers PrimePairVertices.smallTree smallNode7901 :=
  Graph.node_covers _ 7901 smallRow7901 smallNode7883 smallNode7907
    smallNode7883_covers smallNode7907_covers (by decide) (by decide) smallRow7901_covers

def smallNode7877 : Graph := .node 7877 smallRow7877 smallNode7853 smallNode7901
set_option maxRecDepth 4096 in
theorem smallNode7877_covers : Graph.Covers PrimePairVertices.smallTree smallNode7877 :=
  Graph.node_covers _ 7877 smallRow7877 smallNode7853 smallNode7901
    smallNode7853_covers smallNode7901_covers (by decide) (by decide) smallRow7877_covers

def smallNode7927 : Graph := .node 7927 smallRow7927 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7927_covers : Graph.Covers PrimePairVertices.smallTree smallNode7927 :=
  Graph.node_covers _ 7927 smallRow7927 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7927_covers

def smallNode7933 : Graph := .node 7933 smallRow7933 smallNode7927 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7933_covers : Graph.Covers PrimePairVertices.smallTree smallNode7933 :=
  Graph.node_covers _ 7933 smallRow7933 smallNode7927 smallEmpty
    smallNode7927_covers smallEmpty_covers (by decide) (by decide) smallRow7933_covers

def smallNode7949 : Graph := .node 7949 smallRow7949 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7949_covers : Graph.Covers PrimePairVertices.smallTree smallNode7949 :=
  Graph.node_covers _ 7949 smallRow7949 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7949_covers

def smallNode7937 : Graph := .node 7937 smallRow7937 smallNode7933 smallNode7949
set_option maxRecDepth 4096 in
theorem smallNode7937_covers : Graph.Covers PrimePairVertices.smallTree smallNode7937 :=
  Graph.node_covers _ 7937 smallRow7937 smallNode7933 smallNode7949
    smallNode7933_covers smallNode7949_covers (by decide) (by decide) smallRow7937_covers

def smallNode7963 : Graph := .node 7963 smallRow7963 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7963_covers : Graph.Covers PrimePairVertices.smallTree smallNode7963 :=
  Graph.node_covers _ 7963 smallRow7963 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow7963_covers

def smallNode7993 : Graph := .node 7993 smallRow7993 smallNode7963 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7993_covers : Graph.Covers PrimePairVertices.smallTree smallNode7993 :=
  Graph.node_covers _ 7993 smallRow7993 smallNode7963 smallEmpty
    smallNode7963_covers smallEmpty_covers (by decide) (by decide) smallRow7993_covers

def smallNode8011 : Graph := .node 8011 smallRow8011 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8011_covers : Graph.Covers PrimePairVertices.smallTree smallNode8011 :=
  Graph.node_covers _ 8011 smallRow8011 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8011_covers

def smallNode8009 : Graph := .node 8009 smallRow8009 smallNode7993 smallNode8011
set_option maxRecDepth 4096 in
theorem smallNode8009_covers : Graph.Covers PrimePairVertices.smallTree smallNode8009 :=
  Graph.node_covers _ 8009 smallRow8009 smallNode7993 smallNode8011
    smallNode7993_covers smallNode8011_covers (by decide) (by decide) smallRow8009_covers

def smallNode7951 : Graph := .node 7951 smallRow7951 smallNode7937 smallNode8009
set_option maxRecDepth 4096 in
theorem smallNode7951_covers : Graph.Covers PrimePairVertices.smallTree smallNode7951 :=
  Graph.node_covers _ 7951 smallRow7951 smallNode7937 smallNode8009
    smallNode7937_covers smallNode8009_covers (by decide) (by decide) smallRow7951_covers

def smallNode7919 : Graph := .node 7919 smallRow7919 smallNode7877 smallNode7951
set_option maxRecDepth 4096 in
theorem smallNode7919_covers : Graph.Covers PrimePairVertices.smallTree smallNode7919 :=
  Graph.node_covers _ 7919 smallRow7919 smallNode7877 smallNode7951
    smallNode7877_covers smallNode7951_covers (by decide) (by decide) smallRow7919_covers

def smallNode7823 : Graph := .node 7823 smallRow7823 smallNode7717 smallNode7919
set_option maxRecDepth 4096 in
theorem smallNode7823_covers : Graph.Covers PrimePairVertices.smallTree smallNode7823 :=
  Graph.node_covers _ 7823 smallRow7823 smallNode7717 smallNode7919
    smallNode7717_covers smallNode7919_covers (by decide) (by decide) smallRow7823_covers

def smallNode8039 : Graph := .node 8039 smallRow8039 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8039_covers : Graph.Covers PrimePairVertices.smallTree smallNode8039 :=
  Graph.node_covers _ 8039 smallRow8039 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8039_covers

def smallNode8053 : Graph := .node 8053 smallRow8053 smallNode8039 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8053_covers : Graph.Covers PrimePairVertices.smallTree smallNode8053 :=
  Graph.node_covers _ 8053 smallRow8053 smallNode8039 smallEmpty
    smallNode8039_covers smallEmpty_covers (by decide) (by decide) smallRow8053_covers

def smallNode8069 : Graph := .node 8069 smallRow8069 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8069_covers : Graph.Covers PrimePairVertices.smallTree smallNode8069 :=
  Graph.node_covers _ 8069 smallRow8069 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8069_covers

def smallNode8081 : Graph := .node 8081 smallRow8081 smallNode8069 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8081_covers : Graph.Covers PrimePairVertices.smallTree smallNode8081 :=
  Graph.node_covers _ 8081 smallRow8081 smallNode8069 smallEmpty
    smallNode8069_covers smallEmpty_covers (by decide) (by decide) smallRow8081_covers

def smallNode8059 : Graph := .node 8059 smallRow8059 smallNode8053 smallNode8081
set_option maxRecDepth 4096 in
theorem smallNode8059_covers : Graph.Covers PrimePairVertices.smallTree smallNode8059 :=
  Graph.node_covers _ 8059 smallRow8059 smallNode8053 smallNode8081
    smallNode8053_covers smallNode8081_covers (by decide) (by decide) smallRow8059_covers

def smallNode8089 : Graph := .node 8089 smallRow8089 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8089_covers : Graph.Covers PrimePairVertices.smallTree smallNode8089 :=
  Graph.node_covers _ 8089 smallRow8089 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8089_covers

def smallNode8093 : Graph := .node 8093 smallRow8093 smallNode8089 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8093_covers : Graph.Covers PrimePairVertices.smallTree smallNode8093 :=
  Graph.node_covers _ 8093 smallRow8093 smallNode8089 smallEmpty
    smallNode8089_covers smallEmpty_covers (by decide) (by decide) smallRow8093_covers

def smallNode8111 : Graph := .node 8111 smallRow8111 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8111_covers : Graph.Covers PrimePairVertices.smallTree smallNode8111 :=
  Graph.node_covers _ 8111 smallRow8111 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8111_covers

def smallNode8101 : Graph := .node 8101 smallRow8101 smallNode8093 smallNode8111
set_option maxRecDepth 4096 in
theorem smallNode8101_covers : Graph.Covers PrimePairVertices.smallTree smallNode8101 :=
  Graph.node_covers _ 8101 smallRow8101 smallNode8093 smallNode8111
    smallNode8093_covers smallNode8111_covers (by decide) (by decide) smallRow8101_covers

def smallNode8087 : Graph := .node 8087 smallRow8087 smallNode8059 smallNode8101
set_option maxRecDepth 4096 in
theorem smallNode8087_covers : Graph.Covers PrimePairVertices.smallTree smallNode8087 :=
  Graph.node_covers _ 8087 smallRow8087 smallNode8059 smallNode8101
    smallNode8059_covers smallNode8101_covers (by decide) (by decide) smallRow8087_covers

def smallNode8123 : Graph := .node 8123 smallRow8123 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8123_covers : Graph.Covers PrimePairVertices.smallTree smallNode8123 :=
  Graph.node_covers _ 8123 smallRow8123 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8123_covers

def smallNode8147 : Graph := .node 8147 smallRow8147 smallNode8123 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8147_covers : Graph.Covers PrimePairVertices.smallTree smallNode8147 :=
  Graph.node_covers _ 8147 smallRow8147 smallNode8123 smallEmpty
    smallNode8123_covers smallEmpty_covers (by decide) (by decide) smallRow8147_covers

def smallNode8167 : Graph := .node 8167 smallRow8167 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8167_covers : Graph.Covers PrimePairVertices.smallTree smallNode8167 :=
  Graph.node_covers _ 8167 smallRow8167 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8167_covers

def smallNode8161 : Graph := .node 8161 smallRow8161 smallNode8147 smallNode8167
set_option maxRecDepth 4096 in
theorem smallNode8161_covers : Graph.Covers PrimePairVertices.smallTree smallNode8161 :=
  Graph.node_covers _ 8161 smallRow8161 smallNode8147 smallNode8167
    smallNode8147_covers smallNode8167_covers (by decide) (by decide) smallRow8161_covers

def smallNode8179 : Graph := .node 8179 smallRow8179 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8179_covers : Graph.Covers PrimePairVertices.smallTree smallNode8179 :=
  Graph.node_covers _ 8179 smallRow8179 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8179_covers

def smallNode8191 : Graph := .node 8191 smallRow8191 smallNode8179 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8191_covers : Graph.Covers PrimePairVertices.smallTree smallNode8191 :=
  Graph.node_covers _ 8191 smallRow8191 smallNode8179 smallEmpty
    smallNode8179_covers smallEmpty_covers (by decide) (by decide) smallRow8191_covers

def smallNode8219 : Graph := .node 8219 smallRow8219 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8219_covers : Graph.Covers PrimePairVertices.smallTree smallNode8219 :=
  Graph.node_covers _ 8219 smallRow8219 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8219_covers

def smallNode8209 : Graph := .node 8209 smallRow8209 smallNode8191 smallNode8219
set_option maxRecDepth 4096 in
theorem smallNode8209_covers : Graph.Covers PrimePairVertices.smallTree smallNode8209 :=
  Graph.node_covers _ 8209 smallRow8209 smallNode8191 smallNode8219
    smallNode8191_covers smallNode8219_covers (by decide) (by decide) smallRow8209_covers

def smallNode8171 : Graph := .node 8171 smallRow8171 smallNode8161 smallNode8209
set_option maxRecDepth 4096 in
theorem smallNode8171_covers : Graph.Covers PrimePairVertices.smallTree smallNode8171 :=
  Graph.node_covers _ 8171 smallRow8171 smallNode8161 smallNode8209
    smallNode8161_covers smallNode8209_covers (by decide) (by decide) smallRow8171_covers

def smallNode8117 : Graph := .node 8117 smallRow8117 smallNode8087 smallNode8171
set_option maxRecDepth 4096 in
theorem smallNode8117_covers : Graph.Covers PrimePairVertices.smallTree smallNode8117 :=
  Graph.node_covers _ 8117 smallRow8117 smallNode8087 smallNode8171
    smallNode8087_covers smallNode8171_covers (by decide) (by decide) smallRow8117_covers

def smallNode8231 : Graph := .node 8231 smallRow8231 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8231_covers : Graph.Covers PrimePairVertices.smallTree smallNode8231 :=
  Graph.node_covers _ 8231 smallRow8231 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8231_covers

def smallNode8233 : Graph := .node 8233 smallRow8233 smallNode8231 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8233_covers : Graph.Covers PrimePairVertices.smallTree smallNode8233 :=
  Graph.node_covers _ 8233 smallRow8233 smallNode8231 smallEmpty
    smallNode8231_covers smallEmpty_covers (by decide) (by decide) smallRow8233_covers

def smallNode8243 : Graph := .node 8243 smallRow8243 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8243_covers : Graph.Covers PrimePairVertices.smallTree smallNode8243 :=
  Graph.node_covers _ 8243 smallRow8243 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8243_covers

def smallNode8237 : Graph := .node 8237 smallRow8237 smallNode8233 smallNode8243
set_option maxRecDepth 4096 in
theorem smallNode8237_covers : Graph.Covers PrimePairVertices.smallTree smallNode8237 :=
  Graph.node_covers _ 8237 smallRow8237 smallNode8233 smallNode8243
    smallNode8233_covers smallNode8243_covers (by decide) (by decide) smallRow8237_covers

def smallNode8269 : Graph := .node 8269 smallRow8269 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8269_covers : Graph.Covers PrimePairVertices.smallTree smallNode8269 :=
  Graph.node_covers _ 8269 smallRow8269 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8269_covers

def smallNode8273 : Graph := .node 8273 smallRow8273 smallNode8269 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8273_covers : Graph.Covers PrimePairVertices.smallTree smallNode8273 :=
  Graph.node_covers _ 8273 smallRow8273 smallNode8269 smallEmpty
    smallNode8269_covers smallEmpty_covers (by decide) (by decide) smallRow8273_covers

def smallNode8291 : Graph := .node 8291 smallRow8291 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8291_covers : Graph.Covers PrimePairVertices.smallTree smallNode8291 :=
  Graph.node_covers _ 8291 smallRow8291 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8291_covers

def smallNode8287 : Graph := .node 8287 smallRow8287 smallNode8273 smallNode8291
set_option maxRecDepth 4096 in
theorem smallNode8287_covers : Graph.Covers PrimePairVertices.smallTree smallNode8287 :=
  Graph.node_covers _ 8287 smallRow8287 smallNode8273 smallNode8291
    smallNode8273_covers smallNode8291_covers (by decide) (by decide) smallRow8287_covers

def smallNode8263 : Graph := .node 8263 smallRow8263 smallNode8237 smallNode8287
set_option maxRecDepth 4096 in
theorem smallNode8263_covers : Graph.Covers PrimePairVertices.smallTree smallNode8263 :=
  Graph.node_covers _ 8263 smallRow8263 smallNode8237 smallNode8287
    smallNode8237_covers smallNode8287_covers (by decide) (by decide) smallRow8263_covers

def smallNode8297 : Graph := .node 8297 smallRow8297 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8297_covers : Graph.Covers PrimePairVertices.smallTree smallNode8297 :=
  Graph.node_covers _ 8297 smallRow8297 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8297_covers

def smallNode8311 : Graph := .node 8311 smallRow8311 smallNode8297 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8311_covers : Graph.Covers PrimePairVertices.smallTree smallNode8311 :=
  Graph.node_covers _ 8311 smallRow8311 smallNode8297 smallEmpty
    smallNode8297_covers smallEmpty_covers (by decide) (by decide) smallRow8311_covers

def smallNode8329 : Graph := .node 8329 smallRow8329 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8329_covers : Graph.Covers PrimePairVertices.smallTree smallNode8329 :=
  Graph.node_covers _ 8329 smallRow8329 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8329_covers

def smallNode8317 : Graph := .node 8317 smallRow8317 smallNode8311 smallNode8329
set_option maxRecDepth 4096 in
theorem smallNode8317_covers : Graph.Covers PrimePairVertices.smallTree smallNode8317 :=
  Graph.node_covers _ 8317 smallRow8317 smallNode8311 smallNode8329
    smallNode8311_covers smallNode8329_covers (by decide) (by decide) smallRow8317_covers

def smallNode8363 : Graph := .node 8363 smallRow8363 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8363_covers : Graph.Covers PrimePairVertices.smallTree smallNode8363 :=
  Graph.node_covers _ 8363 smallRow8363 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8363_covers

def smallNode8369 : Graph := .node 8369 smallRow8369 smallNode8363 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8369_covers : Graph.Covers PrimePairVertices.smallTree smallNode8369 :=
  Graph.node_covers _ 8369 smallRow8369 smallNode8363 smallEmpty
    smallNode8363_covers smallEmpty_covers (by decide) (by decide) smallRow8369_covers

def smallNode8387 : Graph := .node 8387 smallRow8387 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode8387_covers : Graph.Covers PrimePairVertices.smallTree smallNode8387 :=
  Graph.node_covers _ 8387 smallRow8387 smallEmpty smallEmpty
    smallEmpty_covers smallEmpty_covers (by decide) (by decide) smallRow8387_covers

def smallNode8377 : Graph := .node 8377 smallRow8377 smallNode8369 smallNode8387
set_option maxRecDepth 4096 in
theorem smallNode8377_covers : Graph.Covers PrimePairVertices.smallTree smallNode8377 :=
  Graph.node_covers _ 8377 smallRow8377 smallNode8369 smallNode8387
    smallNode8369_covers smallNode8387_covers (by decide) (by decide) smallRow8377_covers

def smallNode8353 : Graph := .node 8353 smallRow8353 smallNode8317 smallNode8377
set_option maxRecDepth 4096 in
theorem smallNode8353_covers : Graph.Covers PrimePairVertices.smallTree smallNode8353 :=
  Graph.node_covers _ 8353 smallRow8353 smallNode8317 smallNode8377
    smallNode8317_covers smallNode8377_covers (by decide) (by decide) smallRow8353_covers

def smallNode8293 : Graph := .node 8293 smallRow8293 smallNode8263 smallNode8353
set_option maxRecDepth 4096 in
theorem smallNode8293_covers : Graph.Covers PrimePairVertices.smallTree smallNode8293 :=
  Graph.node_covers _ 8293 smallRow8293 smallNode8263 smallNode8353
    smallNode8263_covers smallNode8353_covers (by decide) (by decide) smallRow8293_covers

def smallNode8221 : Graph := .node 8221 smallRow8221 smallNode8117 smallNode8293
set_option maxRecDepth 4096 in
theorem smallNode8221_covers : Graph.Covers PrimePairVertices.smallTree smallNode8221 :=
  Graph.node_covers _ 8221 smallRow8221 smallNode8117 smallNode8293
    smallNode8117_covers smallNode8293_covers (by decide) (by decide) smallRow8221_covers

def smallNode8017 : Graph := .node 8017 smallRow8017 smallNode7823 smallNode8221
set_option maxRecDepth 4096 in
theorem smallNode8017_covers : Graph.Covers PrimePairVertices.smallTree smallNode8017 :=
  Graph.node_covers _ 8017 smallRow8017 smallNode7823 smallNode8221
    smallNode7823_covers smallNode8221_covers (by decide) (by decide) smallRow8017_covers

def smallNode7621 : Graph := .node 7621 smallRow7621 smallNode7247 smallNode8017
set_option maxRecDepth 4096 in
theorem smallNode7621_covers : Graph.Covers PrimePairVertices.smallTree smallNode7621 :=
  Graph.node_covers _ 7621 smallRow7621 smallNode7247 smallNode8017
    smallNode7247_covers smallNode8017_covers (by decide) (by decide) smallRow7621_covers

def smallNode6871 : Graph := .node 6871 smallRow6871 smallNode6151 smallNode7621
set_option maxRecDepth 4096 in
theorem smallNode6871_covers : Graph.Covers PrimePairVertices.smallTree smallNode6871 :=
  Graph.node_covers _ 6871 smallRow6871 smallNode6151 smallNode7621
    smallNode6151_covers smallNode7621_covers (by decide) (by decide) smallRow6871_covers

def smallNode5441 : Graph := .node 5441 smallRow5441 smallNode4007 smallNode6871
set_option maxRecDepth 4096 in
theorem smallNode5441_covers : Graph.Covers PrimePairVertices.smallTree smallNode5441 :=
  Graph.node_covers _ 5441 smallRow5441 smallNode4007 smallNode6871
    smallNode4007_covers smallNode6871_covers (by decide) (by decide) smallRow5441_covers

def smallGraph : Graph := smallNode5441
theorem smallGraph_covers : Graph.Covers PrimePairVertices.smallTree smallGraph := smallNode5441_covers
set_option maxRecDepth 4096 in
theorem smallGraph_eq_searchData : smallGraph = SearchData.graph := rfl
set_option maxRecDepth 4096 in
theorem smallGraph_vertices : smallGraph.vertices = PrimePairVertices.small := by
  rw [smallGraph_eq_searchData]
  exact SearchData.vertices
theorem smallGraph_coverage (p : Nat) (member : p ∈ PrimePairVertices.small) : p ∈ smallGraph.vertices := by
  simpa only [smallGraph_vertices] using member

end ProjectEulerSolutions.PrimePairUpper
