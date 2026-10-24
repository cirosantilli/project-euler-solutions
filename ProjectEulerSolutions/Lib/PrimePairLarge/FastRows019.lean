import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row9883 : SortedNatTree.Tree := (.node 12097 (.node 10909 (.node 9973 (.node 9967 .empty .empty) (.node 10567 .empty .empty)) (.node 11161 (.node 10993 .empty .empty) .empty)) (.node 14557 (.node 13297 (.node 12613 .empty .empty) .empty) (.node 15259 (.node 15121 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9883_ordered : SortedNatTree.ordered row9883 = true := by decide
set_option maxRecDepth 4096 in
theorem row9883_values : SortedNatTree.values row9883 = Data.row9883 := rfl
theorem row9883_checked : SortedNatTree.contains row9883 = Data.row9883.contains := by
  rw [SortedNatTree.contains_eq_list row9883 row9883_ordered, row9883_values]
def row9887 : SortedNatTree.Tree := (.node 11261 (.node 10301 (.node 9923 .empty .empty) (.node 10589 .empty .empty)) (.node 12107 (.node 11909 .empty .empty) (.node 15227 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9887_ordered : SortedNatTree.ordered row9887 = true := by decide
set_option maxRecDepth 4096 in
theorem row9887_values : SortedNatTree.values row9887 = Data.row9887 := rfl
theorem row9887_checked : SortedNatTree.contains row9887 = Data.row9887.contains := by
  rw [SortedNatTree.contains_eq_list row9887 row9887_ordered, row9887_values]
def row9901 : SortedNatTree.Tree := (.node 11959 (.node 11119 (.node 10321 (.node 9931 .empty .empty) (.node 10453 .empty .empty)) (.node 11839 (.node 11833 .empty .empty) .empty)) (.node 14401 (.node 13759 (.node 13219 .empty .empty) .empty) (.node 15901 (.node 14887 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9901_ordered : SortedNatTree.ordered row9901 = true := by decide
set_option maxRecDepth 4096 in
theorem row9901_values : SortedNatTree.values row9901 = Data.row9901 := rfl
theorem row9901_checked : SortedNatTree.contains row9901 = Data.row9901.contains := by
  rw [SortedNatTree.contains_eq_list row9901 row9901_ordered, row9901_values]
def row9907 : SortedNatTree.Tree := (.node 13687 (.node 11863 (.node 10771 (.node 9973 .empty .empty) .empty) (.node 12739 .empty .empty)) (.node 15619 (.node 15493 (.node 15259 .empty .empty) .empty) (.node 15937 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9907_ordered : SortedNatTree.ordered row9907 = true := by decide
set_option maxRecDepth 4096 in
theorem row9907_values : SortedNatTree.values row9907 = Data.row9907 := rfl
theorem row9907_checked : SortedNatTree.contains row9907 = Data.row9907.contains := by
  rw [SortedNatTree.contains_eq_list row9907 row9907_ordered, row9907_values]
def row9923 : SortedNatTree.Tree := (.node 11783 (.node 11447 (.node 11393 .empty .empty) (.node 11699 .empty .empty)) (.node 14249 (.node 12911 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row9923_ordered : SortedNatTree.ordered row9923 = true := by decide
set_option maxRecDepth 4096 in
theorem row9923_values : SortedNatTree.values row9923 = Data.row9923 := rfl
theorem row9923_checked : SortedNatTree.contains row9923 = Data.row9923.contains := by
  rw [SortedNatTree.contains_eq_list row9923 row9923_ordered, row9923_values]
def row9929 : SortedNatTree.Tree := (.node 13877 (.node 11261 (.node 11069 .empty .empty) (.node 13331 .empty .empty)) (.node 15299 (.node 14831 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row9929_ordered : SortedNatTree.ordered row9929 = true := by decide
set_option maxRecDepth 4096 in
theorem row9929_values : SortedNatTree.values row9929 = Data.row9929 := rfl
theorem row9929_checked : SortedNatTree.contains row9929 = Data.row9929.contains := by
  rw [SortedNatTree.contains_eq_list row9929 row9929_ordered, row9929_values]
def row9931 : SortedNatTree.Tree := (.node 12451 (.node 11887 (.node 10711 (.node 10453 .empty .empty) (.node 11149 .empty .empty)) (.node 12277 (.node 11941 .empty .empty) .empty)) (.node 13441 (.node 13309 (.node 12739 .empty .empty) .empty) (.node 15913 (.node 13963 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9931_ordered : SortedNatTree.ordered row9931 = true := by decide
set_option maxRecDepth 4096 in
theorem row9931_values : SortedNatTree.values row9931 = Data.row9931 := rfl
theorem row9931_checked : SortedNatTree.contains row9931 = Data.row9931.contains := by
  rw [SortedNatTree.contains_eq_list row9931 row9931_ordered, row9931_values]
def row9941 : SortedNatTree.Tree := (.node 14207 (.node 12107 (.node 11579 .empty .empty) (.node 12473 .empty .empty)) (.node 15761 (.node 15413 .empty .empty) (.node 15803 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9941_ordered : SortedNatTree.ordered row9941 = true := by decide
set_option maxRecDepth 4096 in
theorem row9941_values : SortedNatTree.values row9941 = Data.row9941 := rfl
theorem row9941_checked : SortedNatTree.contains row9941 = Data.row9941.contains := by
  rw [SortedNatTree.contains_eq_list row9941 row9941_ordered, row9941_values]
def row9949 : SortedNatTree.Tree := (.node 12703 (.node 11257 (.node 10639 (.node 10009 .empty .empty) .empty) (.node 12619 .empty .empty)) (.node 15091 (.node 14593 (.node 13687 .empty .empty) .empty) (.node 15139 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9949_ordered : SortedNatTree.ordered row9949 = true := by decide
set_option maxRecDepth 4096 in
theorem row9949_values : SortedNatTree.values row9949 = Data.row9949 := rfl
theorem row9949_checked : SortedNatTree.contains row9949 = Data.row9949.contains := by
  rw [SortedNatTree.contains_eq_list row9949 row9949_ordered, row9949_values]
def row9967 : SortedNatTree.Tree := (.node 13693 (.node 12391 (.node 10837 (.node 10513 .empty .empty) (.node 12109 .empty .empty)) (.node 13513 (.node 13327 .empty .empty) .empty)) (.node 15607 (.node 14281 (.node 13873 .empty .empty) .empty) (.node 16057 (.node 15667 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9967_ordered : SortedNatTree.ordered row9967 = true := by decide
set_option maxRecDepth 4096 in
theorem row9967_values : SortedNatTree.values row9967 = Data.row9967 := rfl
theorem row9967_checked : SortedNatTree.contains row9967 = Data.row9967.contains := by
  rw [SortedNatTree.contains_eq_list row9967 row9967_ordered, row9967_values]
def row9973 : SortedNatTree.Tree := (.node 14107 (.node 12343 (.node 11677 (.node 10459 .empty .empty) .empty) (.node 13417 (.node 13063 .empty .empty) .empty)) (.node 15373 (.node 14827 (.node 14449 .empty .empty) .empty) (.node 15391 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9973_ordered : SortedNatTree.ordered row9973 = true := by decide
set_option maxRecDepth 4096 in
theorem row9973_values : SortedNatTree.values row9973 = Data.row9973 := rfl
theorem row9973_checked : SortedNatTree.contains row9973 = Data.row9973.contains := by
  rw [SortedNatTree.contains_eq_list row9973 row9973_ordered, row9973_values]
def row10007 : SortedNatTree.Tree := (.node 12347 (.node 12281 (.node 10709 .empty .empty) .empty) (.node 15473 (.node 14207 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10007_ordered : SortedNatTree.ordered row10007 = true := by decide
set_option maxRecDepth 4096 in
theorem row10007_values : SortedNatTree.values row10007 = Data.row10007 := rfl
theorem row10007_checked : SortedNatTree.contains row10007 = Data.row10007.contains := by
  rw [SortedNatTree.contains_eq_list row10007 row10007_ordered, row10007_values]
def row10009 : SortedNatTree.Tree := (.node 13831 (.node 11971 (.node 11251 (.node 10333 .empty .empty) .empty) (.node 13477 (.node 12613 .empty .empty) .empty)) (.node 14821 (.node 14653 (.node 14107 .empty .empty) .empty) (.node 15121 (.node 15061 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10009_ordered : SortedNatTree.ordered row10009 = true := by decide
set_option maxRecDepth 4096 in
theorem row10009_values : SortedNatTree.values row10009 = Data.row10009 := rfl
theorem row10009_checked : SortedNatTree.contains row10009 = Data.row10009.contains := by
  rw [SortedNatTree.contains_eq_list row10009 row10009_ordered, row10009_values]
def row10037 : SortedNatTree.Tree := (.node 13043 (.node 11171 (.node 11003 .empty .empty) (.node 11897 .empty .empty)) (.node 13619 (.node 13397 .empty .empty) (.node 15731 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10037_ordered : SortedNatTree.ordered row10037 = true := by decide
set_option maxRecDepth 4096 in
theorem row10037_values : SortedNatTree.values row10037 = Data.row10037 := rfl
theorem row10037_checked : SortedNatTree.contains row10037 = Data.row10037.contains := by
  rw [SortedNatTree.contains_eq_list row10037 row10037_ordered, row10037_values]
def row10039 : SortedNatTree.Tree := (.node 14251 (.node 12703 (.node 11443 (.node 10597 .empty .empty) (.node 11971 .empty .empty)) (.node 14029 (.node 13831 .empty .empty) .empty)) (.node 15667 (.node 14929 (.node 14869 .empty .empty) .empty) (.node 15889 (.node 15727 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10039_ordered : SortedNatTree.ordered row10039 = true := by decide
set_option maxRecDepth 4096 in
theorem row10039_values : SortedNatTree.values row10039 = Data.row10039 := rfl
theorem row10039_checked : SortedNatTree.contains row10039 = Data.row10039.contains := by
  rw [SortedNatTree.contains_eq_list row10039 row10039_ordered, row10039_values]
def row10061 : SortedNatTree.Tree := (.node 13781 (.node 11987 (.node 10181 (.node 10169 .empty .empty) (.node 10463 .empty .empty)) (.node 13457 (.node 12143 .empty .empty) .empty)) (.node 14489 (.node 14243 (.node 13841 .empty .empty) .empty) (.node 15629 (.node 14699 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10061_ordered : SortedNatTree.ordered row10061 = true := by decide
set_option maxRecDepth 4096 in
theorem row10061_values : SortedNatTree.values row10061 = Data.row10061 := rfl
theorem row10061_checked : SortedNatTree.contains row10061 = Data.row10061.contains := by
  rw [SortedNatTree.contains_eq_list row10061 row10061_ordered, row10061_values]
def row10067 : SortedNatTree.Tree := (.node 13121 (.node 11801 (.node 11003 .empty .empty) (.node 11807 .empty .empty)) (.node 15629 (.node 13907 .empty .empty) (.node 15683 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10067_ordered : SortedNatTree.ordered row10067 = true := by decide
set_option maxRecDepth 4096 in
theorem row10067_values : SortedNatTree.values row10067 = Data.row10067 := rfl
theorem row10067_checked : SortedNatTree.contains row10067 = Data.row10067.contains := by
  rw [SortedNatTree.contains_eq_list row10067 row10067_ordered, row10067_values]
def row10069 : SortedNatTree.Tree := (.node 13093 (.node 12301 (.node 11491 (.node 11047 .empty .empty) (.node 11839 .empty .empty)) (.node 12979 (.node 12637 .empty .empty) .empty)) (.node 14071 (.node 13591 (.node 13399 .empty .empty) .empty) (.node 14767 (.node 14653 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10069_ordered : SortedNatTree.ordered row10069 = true := by decide
set_option maxRecDepth 4096 in
theorem row10069_values : SortedNatTree.values row10069 = Data.row10069 := rfl
theorem row10069_checked : SortedNatTree.contains row10069 = Data.row10069.contains := by
  rw [SortedNatTree.contains_eq_list row10069 row10069_ordered, row10069_values]
def row10079 : SortedNatTree.Tree := (.node 12203 (.node 11909 (.node 10847 (.node 10103 .empty .empty) .empty) (.node 12041 (.node 11939 .empty .empty) .empty)) (.node 14321 (.node 14177 (.node 13859 .empty .empty) .empty) (.node 14951 (.node 14657 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10079_ordered : SortedNatTree.ordered row10079 = true := by decide
set_option maxRecDepth 4096 in
theorem row10079_values : SortedNatTree.values row10079 = Data.row10079 := rfl
theorem row10079_checked : SortedNatTree.contains row10079 = Data.row10079.contains := by
  rw [SortedNatTree.contains_eq_list row10079 row10079_ordered, row10079_values]
def row10091 : SortedNatTree.Tree := (.node 12323 (.node 10601 (.node 10589 (.node 10133 .empty .empty) .empty) (.node 10979 .empty .empty)) (.node 13877 (.node 13577 (.node 13151 .empty .empty) .empty) (.node 15473 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10091_ordered : SortedNatTree.ordered row10091 = true := by decide
set_option maxRecDepth 4096 in
theorem row10091_values : SortedNatTree.values row10091 = Data.row10091 := rfl
theorem row10091_checked : SortedNatTree.contains row10091 = Data.row10091.contains := by
  rw [SortedNatTree.contains_eq_list row10091 row10091_ordered, row10091_values]
def row10093 : SortedNatTree.Tree := (.node 14323 (.node 12721 (.node 10789 .empty .empty) (.node 14281 .empty .empty)) (.node 15259 (.node 14851 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10093_ordered : SortedNatTree.ordered row10093 = true := by decide
set_option maxRecDepth 4096 in
theorem row10093_values : SortedNatTree.values row10093 = Data.row10093 := rfl
theorem row10093_checked : SortedNatTree.contains row10093 = Data.row10093.contains := by
  rw [SortedNatTree.contains_eq_list row10093 row10093_ordered, row10093_values]
def row10099 : SortedNatTree.Tree := (.node 13297 (.node 12919 (.node 10831 (.node 10273 .empty .empty) .empty) (.node 13063 .empty .empty)) (.node 14071 (.node 13921 .empty .empty) (.node 15667 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10099_ordered : SortedNatTree.ordered row10099 = true := by decide
set_option maxRecDepth 4096 in
theorem row10099_values : SortedNatTree.values row10099 = Data.row10099 := rfl
theorem row10099_checked : SortedNatTree.contains row10099 = Data.row10099.contains := by
  rw [SortedNatTree.contains_eq_list row10099 row10099_ordered, row10099_values]
def row10103 : SortedNatTree.Tree := (.node 12983 (.node 11597 (.node 10733 .empty .empty) .empty) (.node 14159 (.node 13001 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10103_ordered : SortedNatTree.ordered row10103 = true := by decide
set_option maxRecDepth 4096 in
theorem row10103_values : SortedNatTree.values row10103 = Data.row10103 := rfl
theorem row10103_checked : SortedNatTree.contains row10103 = Data.row10103.contains := by
  rw [SortedNatTree.contains_eq_list row10103 row10103_ordered, row10103_values]
def row10111 : SortedNatTree.Tree := (.node 13099 (.node 11719 (.node 10957 (.node 10159 .empty .empty) (.node 11083 .empty .empty)) (.node 13093 (.node 12967 .empty .empty) .empty)) (.node 14869 (.node 13687 (.node 13219 .empty .empty) (.node 14557 .empty .empty)) (.node 15583 (.node 15493 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10111_ordered : SortedNatTree.ordered row10111 = true := by decide
set_option maxRecDepth 4096 in
theorem row10111_values : SortedNatTree.values row10111 = Data.row10111 := rfl
theorem row10111_checked : SortedNatTree.contains row10111 = Data.row10111.contains := by
  rw [SortedNatTree.contains_eq_list row10111 row10111_ordered, row10111_values]
def row10133 : SortedNatTree.Tree := (.node 12401 (.node 10937 (.node 10631 (.node 10433 .empty .empty) (.node 10799 .empty .empty)) (.node 11801 (.node 11423 .empty .empty) .empty)) (.node 13997 (.node 12689 (.node 12653 .empty .empty) (.node 13907 .empty .empty)) (.node 15497 (.node 14543 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10133_ordered : SortedNatTree.ordered row10133 = true := by decide
set_option maxRecDepth 4096 in
theorem row10133_values : SortedNatTree.values row10133 = Data.row10133 := rfl
theorem row10133_checked : SortedNatTree.contains row10133 = Data.row10133.contains := by
  rw [SortedNatTree.contains_eq_list row10133 row10133_ordered, row10133_values]
def row10139 : SortedNatTree.Tree := (.node 12923 (.node 11447 (.node 11273 (.node 11159 .empty .empty) (.node 11321 .empty .empty)) (.node 12041 (.node 11813 .empty .empty) .empty)) (.node 14771 (.node 13679 (.node 13451 .empty .empty) (.node 14177 .empty .empty)) (.node 15359 (.node 15269 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10139_ordered : SortedNatTree.ordered row10139 = true := by decide
set_option maxRecDepth 4096 in
theorem row10139_values : SortedNatTree.values row10139 = Data.row10139 := rfl
theorem row10139_checked : SortedNatTree.contains row10139 = Data.row10139.contains := by
  rw [SortedNatTree.contains_eq_list row10139 row10139_ordered, row10139_values]
def row10141 : SortedNatTree.Tree := (.node 13249 (.node 11113 (.node 10243 (.node 10159 .empty .empty) (.node 10711 .empty .empty)) (.node 11887 (.node 11821 .empty .empty) (.node 12763 .empty .empty))) (.node 15199 (.node 14737 (.node 14557 .empty .empty) (.node 14887 .empty .empty)) (.node 15601 (.node 15373 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10141_ordered : SortedNatTree.ordered row10141 = true := by decide
set_option maxRecDepth 4096 in
theorem row10141_values : SortedNatTree.values row10141 = Data.row10141 := rfl
theorem row10141_checked : SortedNatTree.contains row10141 = Data.row10141.contains := by
  rw [SortedNatTree.contains_eq_list row10141 row10141_ordered, row10141_values]
def row10151 : SortedNatTree.Tree := (.node 13859 (.node 13109 (.node 11159 (.node 10601 .empty .empty) (.node 12503 .empty .empty)) (.node 13499 (.node 13217 .empty .empty) .empty)) (.node 14657 (.node 14621 (.node 14447 .empty .empty) .empty) (.node 15791 (.node 14951 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10151_ordered : SortedNatTree.ordered row10151 = true := by decide
set_option maxRecDepth 4096 in
theorem row10151_values : SortedNatTree.values row10151 = Data.row10151 := rfl
theorem row10151_checked : SortedNatTree.contains row10151 = Data.row10151.contains := by
  rw [SortedNatTree.contains_eq_list row10151 row10151_ordered, row10151_values]
def row10159 : SortedNatTree.Tree := (.node 12841 (.node 12211 (.node 11317 (.node 10723 .empty .empty) .empty) (.node 12697 .empty .empty)) (.node 14551 (.node 13267 .empty .empty) (.node 15217 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10159_ordered : SortedNatTree.ordered row10159 = true := by decide
set_option maxRecDepth 4096 in
theorem row10159_values : SortedNatTree.values row10159 = Data.row10159 := rfl
theorem row10159_checked : SortedNatTree.contains row10159 = Data.row10159.contains := by
  rw [SortedNatTree.contains_eq_list row10159 row10159_ordered, row10159_values]
def row10163 : SortedNatTree.Tree := (.node 13187 (.node 12203 (.node 11279 .empty .empty) .empty) (.node 15791 (.node 15551 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10163_ordered : SortedNatTree.ordered row10163 = true := by decide
set_option maxRecDepth 4096 in
theorem row10163_values : SortedNatTree.values row10163 = Data.row10163 := rfl
theorem row10163_checked : SortedNatTree.contains row10163 = Data.row10163.contains := by
  rw [SortedNatTree.contains_eq_list row10163 row10163_ordered, row10163_values]
def row10169 : SortedNatTree.Tree := (.node 13619 (.node 11393 (.node 11261 .empty .empty) (.node 11447 .empty .empty)) (.node 14207 (.node 13967 .empty .empty) (.node 14537 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10169_ordered : SortedNatTree.ordered row10169 = true := by decide
set_option maxRecDepth 4096 in
theorem row10169_values : SortedNatTree.values row10169 = Data.row10169 := rfl
theorem row10169_checked : SortedNatTree.contains row10169 = Data.row10169.contains := by
  rw [SortedNatTree.contains_eq_list row10169 row10169_ordered, row10169_values]
def row10177 : SortedNatTree.Tree := (.node 13441 (.node 11923 (.node 11131 .empty .empty) (.node 13063 .empty .empty)) (.node 14083 (.node 13693 .empty .empty) (.node 14767 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10177_ordered : SortedNatTree.ordered row10177 = true := by decide
set_option maxRecDepth 4096 in
theorem row10177_values : SortedNatTree.values row10177 = Data.row10177 := rfl
theorem row10177_checked : SortedNatTree.contains row10177 = Data.row10177.contains := by
  rw [SortedNatTree.contains_eq_list row10177 row10177_ordered, row10177_values]
def row10181 : SortedNatTree.Tree := (.node 12281 (.node 11279 (.node 10343 (.node 10271 .empty .empty) .empty) (.node 11867 (.node 11519 .empty .empty) .empty)) (.node 15053 (.node 14177 (.node 14057 .empty .empty) .empty) (.node 15473 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10181_ordered : SortedNatTree.ordered row10181 = true := by decide
set_option maxRecDepth 4096 in
theorem row10181_values : SortedNatTree.values row10181 = Data.row10181 := rfl
theorem row10181_checked : SortedNatTree.contains row10181 = Data.row10181.contains := by
  rw [SortedNatTree.contains_eq_list row10181 row10181_ordered, row10181_values]
def row10193 : SortedNatTree.Tree := (.node 13469 (.node 12263 (.node 10529 .empty .empty) .empty) (.node 13499 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10193_ordered : SortedNatTree.ordered row10193 = true := by decide
set_option maxRecDepth 4096 in
theorem row10193_values : SortedNatTree.values row10193 = Data.row10193 := rfl
theorem row10193_checked : SortedNatTree.contains row10193 = Data.row10193.contains := by
  rw [SortedNatTree.contains_eq_list row10193 row10193_ordered, row10193_values]
def row10211 : SortedNatTree.Tree := (.node 14411 (.node 13883 (.node 10589 .empty .empty) .empty) (.node 15737 (.node 15101 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10211_ordered : SortedNatTree.ordered row10211 = true := by decide
set_option maxRecDepth 4096 in
theorem row10211_values : SortedNatTree.values row10211 = Data.row10211 := rfl
theorem row10211_checked : SortedNatTree.contains row10211 = Data.row10211.contains := by
  rw [SortedNatTree.contains_eq_list row10211 row10211_ordered, row10211_values]
def row10223 : SortedNatTree.Tree := (.node 12539 (.node 11789 (.node 10859 (.node 10847 (.node 10733 .empty .empty) .empty) (.node 11447 .empty .empty)) (.node 12107 (.node 11987 .empty .empty) (.node 12113 .empty .empty))) (.node 14519 (.node 13163 (.node 12821 .empty .empty) (.node 13217 .empty .empty)) (.node 14897 (.node 14639 .empty .empty) (.node 15527 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row10223_ordered : SortedNatTree.ordered row10223 = true := by decide
set_option maxRecDepth 4096 in
theorem row10223_values : SortedNatTree.values row10223 = Data.row10223 := rfl
theorem row10223_checked : SortedNatTree.contains row10223 = Data.row10223.contains := by
  rw [SortedNatTree.contains_eq_list row10223 row10223_ordered, row10223_values]
def row10243 : SortedNatTree.Tree := (.node 13183 (.node 10663 (.node 10477 (.node 10303 .empty .empty) .empty) (.node 11731 (.node 11311 .empty .empty) .empty)) (.node 15139 (.node 14419 (.node 13627 .empty .empty) .empty) (.node 15679 (.node 15277 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10243_ordered : SortedNatTree.ordered row10243 = true := by decide
set_option maxRecDepth 4096 in
theorem row10243_values : SortedNatTree.values row10243 = Data.row10243 := rfl
theorem row10243_checked : SortedNatTree.contains row10243 = Data.row10243.contains := by
  rw [SortedNatTree.contains_eq_list row10243 row10243_ordered, row10243_values]
def row10247 : SortedNatTree.Tree := (.node 12917 (.node 12527 (.node 11279 (.node 10457 .empty .empty) .empty) (.node 12641 .empty .empty)) (.node 15161 (.node 14057 .empty .empty) (.node 15383 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10247_ordered : SortedNatTree.ordered row10247 = true := by decide
set_option maxRecDepth 4096 in
theorem row10247_values : SortedNatTree.values row10247 = Data.row10247 := rfl
theorem row10247_checked : SortedNatTree.contains row10247 = Data.row10247.contains := by
  rw [SortedNatTree.contains_eq_list row10247 row10247_ordered, row10247_values]
def row10253 : SortedNatTree.Tree := (.node 13997 (.node 11069 (.node 10529 (.node 10463 .empty .empty) .empty) (.node 12203 .empty .empty)) (.node 15131 (.node 14831 (.node 14177 .empty .empty) .empty) (.node 15527 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10253_ordered : SortedNatTree.ordered row10253 = true := by decide
set_option maxRecDepth 4096 in
theorem row10253_values : SortedNatTree.values row10253 = Data.row10253 := rfl
theorem row10253_checked : SortedNatTree.contains row10253 = Data.row10253.contains := by
  rw [SortedNatTree.contains_eq_list row10253 row10253_ordered, row10253_values]
def row10259 : SortedNatTree.Tree := (.node 14159 (.node 12893 (.node 12101 (.node 10799 .empty .empty) .empty) (.node 13121 (.node 12983 .empty .empty) .empty)) (.node 15131 (.node 15083 (.node 14321 .empty .empty) .empty) (.node 15383 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10259_ordered : SortedNatTree.ordered row10259 = true := by decide
set_option maxRecDepth 4096 in
theorem row10259_values : SortedNatTree.values row10259 = Data.row10259 := rfl
theorem row10259_checked : SortedNatTree.contains row10259 = Data.row10259.contains := by
  rw [SortedNatTree.contains_eq_list row10259 row10259_ordered, row10259_values]
def row10267 : SortedNatTree.Tree := (.node 12409 (.node 11593 (.node 11251 (.node 10837 .empty .empty) .empty) (.node 12097 .empty .empty)) (.node 15061 (.node 14737 (.node 12919 .empty .empty) .empty) (.node 15373 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10267_ordered : SortedNatTree.ordered row10267 = true := by decide
set_option maxRecDepth 4096 in
theorem row10267_values : SortedNatTree.values row10267 = Data.row10267 := rfl
theorem row10267_checked : SortedNatTree.contains row10267 = Data.row10267.contains := by
  rw [SortedNatTree.contains_eq_list row10267 row10267_ordered, row10267_values]
def row10271 : SortedNatTree.Tree := (.node 12641 (.node 11597 (.node 10853 .empty .empty) .empty) (.node 13337 (.node 13127 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10271_ordered : SortedNatTree.ordered row10271 = true := by decide
set_option maxRecDepth 4096 in
theorem row10271_values : SortedNatTree.values row10271 = Data.row10271 := rfl
theorem row10271_checked : SortedNatTree.contains row10271 = Data.row10271.contains := by
  rw [SortedNatTree.contains_eq_list row10271 row10271_ordered, row10271_values]
def row10273 : SortedNatTree.Tree := (.node 12109 (.node 10987 (.node 10303 .empty .empty) (.node 11731 .empty .empty)) (.node 15277 (.node 13297 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10273_ordered : SortedNatTree.ordered row10273 = true := by decide
set_option maxRecDepth 4096 in
theorem row10273_values : SortedNatTree.values row10273 = Data.row10273 := rfl
theorem row10273_checked : SortedNatTree.contains row10273 = Data.row10273.contains := by
  rw [SortedNatTree.contains_eq_list row10273 row10273_ordered, row10273_values]
def row10289 : SortedNatTree.Tree := (.node 12473 (.node 11069 (.node 10859 (.node 10427 .empty .empty) .empty) (.node 11657 (.node 11489 .empty .empty) .empty)) (.node 14207 (.node 13799 (.node 13103 .empty .empty) .empty) (.node 14303 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10289_ordered : SortedNatTree.ordered row10289 = true := by decide
set_option maxRecDepth 4096 in
theorem row10289_values : SortedNatTree.values row10289 = Data.row10289 := rfl
theorem row10289_checked : SortedNatTree.contains row10289 = Data.row10289.contains := by
  rw [SortedNatTree.contains_eq_list row10289 row10289_ordered, row10289_values]
def row10301 : SortedNatTree.Tree := (.node 12527 (.node 11717 (.node 10889 (.node 10343 .empty .empty) .empty) (.node 11933 .empty .empty)) (.node 13697 (.node 12899 (.node 12641 .empty .empty) .empty) (.node 14867 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10301_ordered : SortedNatTree.ordered row10301 = true := by decide
set_option maxRecDepth 4096 in
theorem row10301_values : SortedNatTree.values row10301 = Data.row10301 := rfl
theorem row10301_checked : SortedNatTree.contains row10301 = Data.row10301.contains := by
  rw [SortedNatTree.contains_eq_list row10301 row10301_ordered, row10301_values]
def row10303 : SortedNatTree.Tree := (.node 12721 (.node 11551 (.node 11083 (.node 10861 .empty .empty) (.node 11131 .empty .empty)) (.node 11941 (.node 11719 .empty .empty) .empty)) (.node 13831 (.node 13759 (.node 13327 .empty .empty) .empty) (.node 14851 (.node 14533 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10303_ordered : SortedNatTree.ordered row10303 = true := by decide
set_option maxRecDepth 4096 in
theorem row10303_values : SortedNatTree.values row10303 = Data.row10303 := rfl
theorem row10303_checked : SortedNatTree.contains row10303 = Data.row10303.contains := by
  rw [SortedNatTree.contains_eq_list row10303 row10303_ordered, row10303_values]
def row10313 : SortedNatTree.Tree := (.node 13757 (.node 11399 (.node 10799 (.node 10391 .empty .empty) .empty) (.node 12437 (.node 11657 .empty .empty) .empty)) (.node 15083 (.node 14759 (.node 14087 .empty .empty) .empty) (.node 15629 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10313_ordered : SortedNatTree.ordered row10313 = true := by decide
set_option maxRecDepth 4096 in
theorem row10313_values : SortedNatTree.values row10313 = Data.row10313 := rfl
theorem row10313_checked : SortedNatTree.contains row10313 = Data.row10313.contains := by
  rw [SortedNatTree.contains_eq_list row10313 row10313_ordered, row10313_values]
def row10321 : SortedNatTree.Tree := (.node 14107 (.node 12973 (.node 12457 (.node 11287 .empty .empty) .empty) (.node 13513 .empty .empty)) (.node 15241 (.node 15187 .empty .empty) (.node 15619 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10321_ordered : SortedNatTree.ordered row10321 = true := by decide
set_option maxRecDepth 4096 in
theorem row10321_values : SortedNatTree.values row10321 = Data.row10321 := rfl
theorem row10321_checked : SortedNatTree.contains row10321 = Data.row10321.contains := by
  rw [SortedNatTree.contains_eq_list row10321 row10321_ordered, row10321_values]
def row10331 : SortedNatTree.Tree := (.node 12893 (.node 11489 (.node 11411 (.node 10499 .empty .empty) .empty) (.node 12347 .empty .empty)) (.node 14081 (.node 13841 (.node 12959 .empty .empty) .empty) (.node 14891 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10331_ordered : SortedNatTree.ordered row10331 = true := by decide
set_option maxRecDepth 4096 in
theorem row10331_values : SortedNatTree.values row10331 = Data.row10331 := rfl
theorem row10331_checked : SortedNatTree.contains row10331 = Data.row10331.contains := by
  rw [SortedNatTree.contains_eq_list row10331 row10331_ordered, row10331_values]
def row10333 : SortedNatTree.Tree := (.node 13513 (.node 11311 (.node 11047 (.node 10939 .empty .empty) .empty) (.node 13291 (.node 12979 .empty .empty) .empty)) (.node 14461 (.node 14407 (.node 13789 .empty .empty) .empty) (.node 15541 (.node 15091 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10333_ordered : SortedNatTree.ordered row10333 = true := by decide
set_option maxRecDepth 4096 in
theorem row10333_values : SortedNatTree.values row10333 = Data.row10333 := rfl
theorem row10333_checked : SortedNatTree.contains row10333 = Data.row10333.contains := by
  rw [SortedNatTree.contains_eq_list row10333 row10333_ordered, row10333_values]
def row10337 : SortedNatTree.Tree := (.node 13697 (.node 11681 (.node 11579 (.node 11519 .empty .empty) .empty) (.node 13049 .empty .empty)) (.node 13931 (.node 13763 (.node 13757 .empty .empty) .empty) (.node 15017 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10337_ordered : SortedNatTree.ordered row10337 = true := by decide
set_option maxRecDepth 4096 in
theorem row10337_values : SortedNatTree.values row10337 = Data.row10337 := rfl
theorem row10337_checked : SortedNatTree.contains row10337 = Data.row10337.contains := by
  rw [SortedNatTree.contains_eq_list row10337 row10337_ordered, row10337_values]
def row10343 : SortedNatTree.Tree := (.node 11717 (.node 11273 (.node 11177 (.node 10601 .empty .empty) .empty) (.node 11351 .empty .empty)) (.node 14627 (.node 13829 (.node 12659 .empty .empty) .empty) (.node 15473 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10343_ordered : SortedNatTree.ordered row10343 = true := by decide
set_option maxRecDepth 4096 in
theorem row10343_values : SortedNatTree.values row10343 = Data.row10343 := rfl
theorem row10343_checked : SortedNatTree.contains row10343 = Data.row10343.contains := by
  rw [SortedNatTree.contains_eq_list row10343 row10343_ordered, row10343_values]
def row10357 : SortedNatTree.Tree := (.node 12967 (.node 11959 (.node 11731 (.node 11239 .empty .empty) .empty) (.node 12289 (.node 12157 .empty .empty) .empty)) (.node 15349 (.node 14731 (.node 13033 .empty .empty) .empty) (.node 15511 (.node 15427 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10357_ordered : SortedNatTree.ordered row10357 = true := by decide
set_option maxRecDepth 4096 in
theorem row10357_values : SortedNatTree.values row10357 = Data.row10357 := rfl
theorem row10357_checked : SortedNatTree.contains row10357 = Data.row10357.contains := by
  rw [SortedNatTree.contains_eq_list row10357 row10357_ordered, row10357_values]
def row10369 : SortedNatTree.Tree := (.node 13873 (.node 13177 (.node 12577 (.node 11119 .empty .empty) .empty) (.node 13723 .empty .empty)) (.node 14653 (.node 14479 .empty .empty) (.node 14731 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10369_ordered : SortedNatTree.ordered row10369 = true := by decide
set_option maxRecDepth 4096 in
theorem row10369_values : SortedNatTree.values row10369 = Data.row10369 := rfl
theorem row10369_checked : SortedNatTree.contains row10369 = Data.row10369.contains := by
  rw [SortedNatTree.contains_eq_list row10369 row10369_ordered, row10369_values]
def row10391 : SortedNatTree.Tree := (.node 11897 (.node 11399 .empty .empty) (.node 14969 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10391_ordered : SortedNatTree.ordered row10391 = true := by decide
set_option maxRecDepth 4096 in
theorem row10391_values : SortedNatTree.values row10391 = Data.row10391 := rfl
theorem row10391_checked : SortedNatTree.contains row10391 = Data.row10391.contains := by
  rw [SortedNatTree.contains_eq_list row10391 row10391_ordered, row10391_values]
def row10399 : SortedNatTree.Tree := (.node 13249 (.node 11329 (.node 10909 .empty .empty) (.node 12619 .empty .empty)) (.node 14851 (.node 13669 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10399_ordered : SortedNatTree.ordered row10399 = true := by decide
set_option maxRecDepth 4096 in
theorem row10399_values : SortedNatTree.values row10399 = Data.row10399 := rfl
theorem row10399_checked : SortedNatTree.contains row10399 = Data.row10399.contains := by
  rw [SortedNatTree.contains_eq_list row10399 row10399_ordered, row10399_values]
def row10427 : SortedNatTree.Tree := (.node 13229 (.node 12401 (.node 10667 .empty .empty) (.node 12689 .empty .empty)) (.node 14627 (.node 13781 .empty .empty) (.node 14753 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10427_ordered : SortedNatTree.ordered row10427 = true := by decide
set_option maxRecDepth 4096 in
theorem row10427_values : SortedNatTree.values row10427 = Data.row10427 := rfl
theorem row10427_checked : SortedNatTree.contains row10427 = Data.row10427.contains := by
  rw [SortedNatTree.contains_eq_list row10427 row10427_ordered, row10427_values]
def row10429 : SortedNatTree.Tree := (.node 13537 (.node 12511 (.node 11887 (.node 11503 .empty .empty) .empty) (.node 13381 .empty .empty)) (.node 14683 (.node 14083 .empty .empty) (.node 14797 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10429_ordered : SortedNatTree.ordered row10429 = true := by decide
set_option maxRecDepth 4096 in
theorem row10429_values : SortedNatTree.values row10429 = Data.row10429 := rfl
theorem row10429_checked : SortedNatTree.contains row10429 = Data.row10429.contains := by
  rw [SortedNatTree.contains_eq_list row10429 row10429_ordered, row10429_values]
def row10433 : SortedNatTree.Tree := (.node 13709 (.node 12041 (.node 10781 .empty .empty) (.node 12713 .empty .empty)) (.node 13967 (.node 13841 .empty .empty) (.node 14549 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10433_ordered : SortedNatTree.ordered row10433 = true := by decide
set_option maxRecDepth 4096 in
theorem row10433_values : SortedNatTree.values row10433 = Data.row10433 := rfl
theorem row10433_checked : SortedNatTree.contains row10433 = Data.row10433.contains := by
  rw [SortedNatTree.contains_eq_list row10433 row10433_ordered, row10433_values]
def row10453 : SortedNatTree.Tree := (.node 12823 (.node 11503 (.node 11329 (.node 10477 .empty .empty) .empty) (.node 12553 (.node 12547 .empty .empty) .empty)) (.node 13723 (.node 13411 (.node 12973 .empty .empty) .empty) (.node 15451 (.node 14461 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10453_ordered : SortedNatTree.ordered row10453 = true := by decide
set_option maxRecDepth 4096 in
theorem row10453_values : SortedNatTree.values row10453 = Data.row10453 := rfl
theorem row10453_checked : SortedNatTree.contains row10453 = Data.row10453.contains := by
  rw [SortedNatTree.contains_eq_list row10453 row10453_ordered, row10453_values]
def row10457 : SortedNatTree.Tree := (.node 13313 (.node 12377 (.node 11717 (.node 11621 .empty .empty) .empty) (.node 12821 .empty .empty)) (.node 14447 (.node 14207 .empty .empty) (.node 15287 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10457_ordered : SortedNatTree.ordered row10457 = true := by decide
set_option maxRecDepth 4096 in
theorem row10457_values : SortedNatTree.values row10457 = Data.row10457 := rfl
theorem row10457_checked : SortedNatTree.contains row10457 = Data.row10457.contains := by
  rw [SortedNatTree.contains_eq_list row10457 row10457_ordered, row10457_values]
def row10459 : SortedNatTree.Tree := (.node 14083 (.node 13291 (.node 10687 (.node 10501 .empty .empty) (.node 11173 .empty .empty)) (.node 13789 (.node 13597 .empty .empty) .empty)) (.node 14713 (.node 14449 (.node 14197 .empty .empty) .empty) (.node 15511 (.node 14827 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10459_ordered : SortedNatTree.ordered row10459 = true := by decide
set_option maxRecDepth 4096 in
theorem row10459_values : SortedNatTree.values row10459 = Data.row10459 := rfl
theorem row10459_checked : SortedNatTree.contains row10459 = Data.row10459.contains := by
  rw [SortedNatTree.contains_eq_list row10459 row10459_ordered, row10459_values]
def row10463 : SortedNatTree.Tree := (.node 12269 (.node 11807 (.node 11657 (.node 11177 .empty .empty) .empty) (.node 11831 .empty .empty)) (.node 12959 (.node 12671 (.node 12329 .empty .empty) .empty) (.node 14627 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10463_ordered : SortedNatTree.ordered row10463 = true := by decide
set_option maxRecDepth 4096 in
theorem row10463_values : SortedNatTree.values row10463 = Data.row10463 := rfl
theorem row10463_checked : SortedNatTree.contains row10463 = Data.row10463.contains := by
  rw [SortedNatTree.contains_eq_list row10463 row10463_ordered, row10463_values]
def row10477 : SortedNatTree.Tree := (.node 13267 (.node 11551 (.node 11131 (.node 10567 .empty .empty) .empty) (.node 13171 (.node 12097 .empty .empty) .empty)) (.node 13921 (.node 13729 (.node 13441 .empty .empty) .empty) (.node 15439 (.node 15349 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10477_ordered : SortedNatTree.ordered row10477 = true := by decide
set_option maxRecDepth 4096 in
theorem row10477_values : SortedNatTree.values row10477 = Data.row10477 := rfl
theorem row10477_checked : SortedNatTree.contains row10477 = Data.row10477.contains := by
  rw [SortedNatTree.contains_eq_list row10477 row10477_ordered, row10477_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
