import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row11779 : SortedNatTree.Tree := (.node 13591 (.node 12577 (.node 11953 .empty .empty) (.node 13183 .empty .empty)) (.node 13903 (.node 13687 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11779_ordered : SortedNatTree.ordered row11779 = true := by decide
set_option maxRecDepth 4096 in
theorem row11779_values : SortedNatTree.values row11779 = Data.row11779 := rfl
theorem row11779_checked : SortedNatTree.contains row11779 = Data.row11779.contains := by
  rw [SortedNatTree.contains_eq_list row11779 row11779_ordered, row11779_values]
def row11783 : SortedNatTree.Tree := (.node 13421 (.node 11867 .empty .empty) (.node 13463 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11783_ordered : SortedNatTree.ordered row11783 = true := by decide
set_option maxRecDepth 4096 in
theorem row11783_values : SortedNatTree.values row11783 = Data.row11783 := rfl
theorem row11783_checked : SortedNatTree.contains row11783 = Data.row11783.contains := by
  rw [SortedNatTree.contains_eq_list row11783 row11783_ordered, row11783_values]
def row11789 : SortedNatTree.Tree := (.node 12269 (.node 11933 .empty .empty) (.node 13469 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11789_ordered : SortedNatTree.ordered row11789 = true := by decide
set_option maxRecDepth 4096 in
theorem row11789_values : SortedNatTree.values row11789 = Data.row11789 := rfl
theorem row11789_checked : SortedNatTree.contains row11789 = Data.row11789.contains := by
  rw [SortedNatTree.contains_eq_list row11789 row11789_ordered, row11789_values]
def row11801 : SortedNatTree.Tree := (.node 13577 (.node 13001 .empty .empty) (.node 13763 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11801_ordered : SortedNatTree.ordered row11801 = true := by decide
set_option maxRecDepth 4096 in
theorem row11801_values : SortedNatTree.values row11801 = Data.row11801 := rfl
theorem row11801_checked : SortedNatTree.contains row11801 = Data.row11801.contains := by
  rw [SortedNatTree.contains_eq_list row11801 row11801_ordered, row11801_values]
def row11807 : SortedNatTree.Tree := (.node 13799 (.node 13619 (.node 12647 .empty .empty) (.node 13757 .empty .empty)) (.node 14159 (.node 13907 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11807_ordered : SortedNatTree.ordered row11807 = true := by decide
set_option maxRecDepth 4096 in
theorem row11807_values : SortedNatTree.values row11807 = Data.row11807 := rfl
theorem row11807_checked : SortedNatTree.contains row11807 = Data.row11807.contains := by
  rw [SortedNatTree.contains_eq_list row11807 row11807_ordered, row11807_values]
def row11813 : SortedNatTree.Tree := (.node 12953 (.node 12659 (.node 11927 .empty .empty) .empty) (.node 13151 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11813_ordered : SortedNatTree.ordered row11813 = true := by decide
set_option maxRecDepth 4096 in
theorem row11813_values : SortedNatTree.values row11813 = Data.row11813 := rfl
theorem row11813_checked : SortedNatTree.contains row11813 = Data.row11813.contains := by
  rw [SortedNatTree.contains_eq_list row11813 row11813_ordered, row11813_values]
def row11821 : SortedNatTree.Tree := (.node 13711 (.node 13567 (.node 12343 .empty .empty) .empty) (.node 14149 (.node 14107 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11821_ordered : SortedNatTree.ordered row11821 = true := by decide
set_option maxRecDepth 4096 in
theorem row11821_values : SortedNatTree.values row11821 = Data.row11821 := rfl
theorem row11821_checked : SortedNatTree.contains row11821 = Data.row11821.contains := by
  rw [SortedNatTree.contains_eq_list row11821 row11821_ordered, row11821_values]
def row11827 : SortedNatTree.Tree := (.node 12547 (.node 12541 (.node 12379 .empty .empty) .empty) (.node 12823 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11827_ordered : SortedNatTree.ordered row11827 = true := by decide
set_option maxRecDepth 4096 in
theorem row11827_values : SortedNatTree.values row11827 = Data.row11827 := rfl
theorem row11827_checked : SortedNatTree.contains row11827 = Data.row11827.contains := by
  rw [SortedNatTree.contains_eq_list row11827 row11827_ordered, row11827_values]
def row11831 : SortedNatTree.Tree := (.node 13649 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11831_ordered : SortedNatTree.ordered row11831 = true := by decide
set_option maxRecDepth 4096 in
theorem row11831_values : SortedNatTree.values row11831 = Data.row11831 := rfl
theorem row11831_checked : SortedNatTree.contains row11831 = Data.row11831.contains := by
  rw [SortedNatTree.contains_eq_list row11831 row11831_ordered, row11831_values]
def row11833 : SortedNatTree.Tree := (.node 13267 (.node 12379 (.node 12301 .empty .empty) .empty) (.node 14029 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11833_ordered : SortedNatTree.ordered row11833 = true := by decide
set_option maxRecDepth 4096 in
theorem row11833_values : SortedNatTree.values row11833 = Data.row11833 := rfl
theorem row11833_checked : SortedNatTree.contains row11833 = Data.row11833.contains := by
  rw [SortedNatTree.contains_eq_list row11833 row11833_ordered, row11833_values]
def row11839 : SortedNatTree.Tree := (.node 12721 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11839_ordered : SortedNatTree.ordered row11839 = true := by decide
set_option maxRecDepth 4096 in
theorem row11839_values : SortedNatTree.values row11839 = Data.row11839 := rfl
theorem row11839_checked : SortedNatTree.contains row11839 = Data.row11839.contains := by
  rw [SortedNatTree.contains_eq_list row11839 row11839_ordered, row11839_values]
def row11863 : SortedNatTree.Tree := (.node 13903 (.node 12157 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11863_ordered : SortedNatTree.ordered row11863 = true := by decide
set_option maxRecDepth 4096 in
theorem row11863_values : SortedNatTree.values row11863 = Data.row11863 := rfl
theorem row11863_checked : SortedNatTree.contains row11863 = Data.row11863.contains := by
  rw [SortedNatTree.contains_eq_list row11863 row11863_ordered, row11863_values]
def row11867 : SortedNatTree.Tree := (.node 13907 (.node 12113 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11867_ordered : SortedNatTree.ordered row11867 = true := by decide
set_option maxRecDepth 4096 in
theorem row11867_values : SortedNatTree.values row11867 = Data.row11867 := rfl
theorem row11867_checked : SortedNatTree.contains row11867 = Data.row11867.contains := by
  rw [SortedNatTree.contains_eq_list row11867 row11867_ordered, row11867_values]
def row11887 : SortedNatTree.Tree := (.node 14143 (.node 12043 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11887_ordered : SortedNatTree.ordered row11887 = true := by decide
set_option maxRecDepth 4096 in
theorem row11887_values : SortedNatTree.values row11887 = Data.row11887 := rfl
theorem row11887_checked : SortedNatTree.contains row11887 = Data.row11887.contains := by
  rw [SortedNatTree.contains_eq_list row11887 row11887_ordered, row11887_values]
def row11897 : SortedNatTree.Tree := (.node 12527 (.node 12101 .empty .empty) (.node 13781 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11897_ordered : SortedNatTree.ordered row11897 = true := by decide
set_option maxRecDepth 4096 in
theorem row11897_values : SortedNatTree.values row11897 = Data.row11897 := rfl
theorem row11897_checked : SortedNatTree.contains row11897 = Data.row11897.contains := by
  rw [SortedNatTree.contains_eq_list row11897 row11897_ordered, row11897_values]
def row11903 : SortedNatTree.Tree := (.node 12527 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11903_ordered : SortedNatTree.ordered row11903 = true := by decide
set_option maxRecDepth 4096 in
theorem row11903_values : SortedNatTree.values row11903 = Data.row11903 := rfl
theorem row11903_checked : SortedNatTree.contains row11903 = Data.row11903.contains := by
  rw [SortedNatTree.contains_eq_list row11903 row11903_ordered, row11903_values]
def row11909 : SortedNatTree.Tree := (.node 13859 (.node 12473 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11909_ordered : SortedNatTree.ordered row11909 = true := by decide
set_option maxRecDepth 4096 in
theorem row11909_values : SortedNatTree.values row11909 = Data.row11909 := rfl
theorem row11909_checked : SortedNatTree.contains row11909 = Data.row11909.contains := by
  rw [SortedNatTree.contains_eq_list row11909 row11909_ordered, row11909_values]
def row11923 : SortedNatTree.Tree := (.node 12547 (.node 12211 (.node 12007 .empty .empty) .empty) (.node 13729 (.node 12907 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11923_ordered : SortedNatTree.ordered row11923 = true := by decide
set_option maxRecDepth 4096 in
theorem row11923_values : SortedNatTree.values row11923 = Data.row11923 := rfl
theorem row11923_checked : SortedNatTree.contains row11923 = Data.row11923.contains := by
  rw [SortedNatTree.contains_eq_list row11923 row11923_ordered, row11923_values]
def row11927 : SortedNatTree.Tree := (.node 12959 (.node 12923 (.node 11981 .empty .empty) .empty) (.node 13103 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11927_ordered : SortedNatTree.ordered row11927 = true := by decide
set_option maxRecDepth 4096 in
theorem row11927_values : SortedNatTree.values row11927 = Data.row11927 := rfl
theorem row11927_checked : SortedNatTree.contains row11927 = Data.row11927.contains := by
  rw [SortedNatTree.contains_eq_list row11927 row11927_ordered, row11927_values]
def row11933 : SortedNatTree.Tree := (.node 12401 (.node 12251 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11933_ordered : SortedNatTree.ordered row11933 = true := by decide
set_option maxRecDepth 4096 in
theorem row11933_values : SortedNatTree.values row11933 = Data.row11933 := rfl
theorem row11933_checked : SortedNatTree.contains row11933 = Data.row11933.contains := by
  rw [SortedNatTree.contains_eq_list row11933 row11933_ordered, row11933_values]
def row11939 : SortedNatTree.Tree := (.node 13121 (.node 12113 (.node 11981 .empty .empty) (.node 12953 .empty .empty)) (.node 13367 (.node 13229 .empty .empty) (.node 13397 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11939_ordered : SortedNatTree.ordered row11939 = true := by decide
set_option maxRecDepth 4096 in
theorem row11939_values : SortedNatTree.values row11939 = Data.row11939 := rfl
theorem row11939_checked : SortedNatTree.contains row11939 = Data.row11939.contains := by
  rw [SortedNatTree.contains_eq_list row11939 row11939_ordered, row11939_values]
def row11941 : SortedNatTree.Tree := (.node 12889 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11941_ordered : SortedNatTree.ordered row11941 = true := by decide
set_option maxRecDepth 4096 in
theorem row11941_values : SortedNatTree.values row11941 = Data.row11941 := rfl
theorem row11941_checked : SortedNatTree.contains row11941 = Data.row11941.contains := by
  rw [SortedNatTree.contains_eq_list row11941 row11941_ordered, row11941_values]
def row11953 : SortedNatTree.Tree := (.node 13003 (.node 12487 (.node 12163 .empty .empty) .empty) (.node 13789 (.node 13723 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11953_ordered : SortedNatTree.ordered row11953 = true := by decide
set_option maxRecDepth 4096 in
theorem row11953_values : SortedNatTree.values row11953 = Data.row11953 := rfl
theorem row11953_checked : SortedNatTree.contains row11953 = Data.row11953.contains := by
  rw [SortedNatTree.contains_eq_list row11953 row11953_ordered, row11953_values]
def row11959 : SortedNatTree.Tree := .empty
set_option maxRecDepth 4096 in
theorem row11959_ordered : SortedNatTree.ordered row11959 = true := by decide
set_option maxRecDepth 4096 in
theorem row11959_values : SortedNatTree.values row11959 = Data.row11959 := rfl
theorem row11959_checked : SortedNatTree.contains row11959 = Data.row11959.contains := by
  rw [SortedNatTree.contains_eq_list row11959 row11959_ordered, row11959_values]
def row11969 : SortedNatTree.Tree := (.node 13829 (.node 13397 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11969_ordered : SortedNatTree.ordered row11969 = true := by decide
set_option maxRecDepth 4096 in
theorem row11969_values : SortedNatTree.values row11969 = Data.row11969 := rfl
theorem row11969_checked : SortedNatTree.contains row11969 = Data.row11969.contains := by
  rw [SortedNatTree.contains_eq_list row11969 row11969_ordered, row11969_values]
def row11971 : SortedNatTree.Tree := (.node 13681 (.node 13441 (.node 12277 .empty .empty) .empty) (.node 13831 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11971_ordered : SortedNatTree.ordered row11971 = true := by decide
set_option maxRecDepth 4096 in
theorem row11971_values : SortedNatTree.values row11971 = Data.row11971 := rfl
theorem row11971_checked : SortedNatTree.contains row11971 = Data.row11971.contains := by
  rw [SortedNatTree.contains_eq_list row11971 row11971_ordered, row11971_values]
def row11981 : SortedNatTree.Tree := (.node 13691 (.node 12611 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11981_ordered : SortedNatTree.ordered row11981 = true := by decide
set_option maxRecDepth 4096 in
theorem row11981_values : SortedNatTree.values row11981 = Data.row11981 := rfl
theorem row11981_checked : SortedNatTree.contains row11981 = Data.row11981.contains := by
  rw [SortedNatTree.contains_eq_list row11981 row11981_ordered, row11981_values]
def row11987 : SortedNatTree.Tree := (.node 13127 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11987_ordered : SortedNatTree.ordered row11987 = true := by decide
set_option maxRecDepth 4096 in
theorem row11987_values : SortedNatTree.values row11987 = Data.row11987 := rfl
theorem row11987_checked : SortedNatTree.contains row11987 = Data.row11987.contains := by
  rw [SortedNatTree.contains_eq_list row11987 row11987_ordered, row11987_values]
def row12007 : SortedNatTree.Tree := (.node 13309 (.node 13291 (.node 12613 .empty .empty) .empty) (.node 13513 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12007_ordered : SortedNatTree.ordered row12007 = true := by decide
set_option maxRecDepth 4096 in
theorem row12007_values : SortedNatTree.values row12007 = Data.row12007 := rfl
theorem row12007_checked : SortedNatTree.contains row12007 = Data.row12007.contains := by
  rw [SortedNatTree.contains_eq_list row12007 row12007_ordered, row12007_values]
def row12011 : SortedNatTree.Tree := (.node 13553 (.node 12959 .empty .empty) (.node 13649 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12011_ordered : SortedNatTree.ordered row12011 = true := by decide
set_option maxRecDepth 4096 in
theorem row12011_values : SortedNatTree.values row12011 = Data.row12011 := rfl
theorem row12011_checked : SortedNatTree.contains row12011 = Data.row12011.contains := by
  rw [SortedNatTree.contains_eq_list row12011 row12011_ordered, row12011_values]
def row12037 : SortedNatTree.Tree := (.node 13591 (.node 12739 (.node 12241 .empty .empty) .empty) (.node 13633 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12037_ordered : SortedNatTree.ordered row12037 = true := by decide
set_option maxRecDepth 4096 in
theorem row12037_values : SortedNatTree.values row12037 = Data.row12037 := rfl
theorem row12037_checked : SortedNatTree.contains row12037 = Data.row12037.contains := by
  rw [SortedNatTree.contains_eq_list row12037 row12037_ordered, row12037_values]
def row12041 : SortedNatTree.Tree := (.node 13397 (.node 13259 .empty .empty) (.node 13931 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12041_ordered : SortedNatTree.ordered row12041 = true := by decide
set_option maxRecDepth 4096 in
theorem row12041_values : SortedNatTree.values row12041 = Data.row12041 := rfl
theorem row12041_checked : SortedNatTree.contains row12041 = Data.row12041.contains := by
  rw [SortedNatTree.contains_eq_list row12041 row12041_ordered, row12041_values]
def row12043 : SortedNatTree.Tree := .empty
set_option maxRecDepth 4096 in
theorem row12043_ordered : SortedNatTree.ordered row12043 = true := by decide
set_option maxRecDepth 4096 in
theorem row12043_values : SortedNatTree.values row12043 = Data.row12043 := rfl
theorem row12043_checked : SortedNatTree.contains row12043 = Data.row12043.contains := by
  rw [SortedNatTree.contains_eq_list row12043 row12043_ordered, row12043_values]
def row12049 : SortedNatTree.Tree := (.node 12739 .empty .empty)
set_option maxRecDepth 4096 in
theorem row12049_ordered : SortedNatTree.ordered row12049 = true := by decide
set_option maxRecDepth 4096 in
theorem row12049_values : SortedNatTree.values row12049 = Data.row12049 := rfl
theorem row12049_checked : SortedNatTree.contains row12049 = Data.row12049.contains := by
  rw [SortedNatTree.contains_eq_list row12049 row12049_ordered, row12049_values]
def row12071 : SortedNatTree.Tree := (.node 13421 (.node 12497 (.node 12377 (.node 12113 .empty .empty) .empty) (.node 12953 .empty .empty)) (.node 13763 (.node 13709 (.node 13499 .empty .empty) .empty) (.node 13877 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row12071_ordered : SortedNatTree.ordered row12071 = true := by decide
set_option maxRecDepth 4096 in
theorem row12071_values : SortedNatTree.values row12071 = Data.row12071 := rfl
theorem row12071_checked : SortedNatTree.contains row12071 = Data.row12071.contains := by
  rw [SortedNatTree.contains_eq_list row12071 row12071_ordered, row12071_values]
def row12073 : SortedNatTree.Tree := (.node 12703 (.node 12451 (.node 12373 .empty .empty) .empty) (.node 13147 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12073_ordered : SortedNatTree.ordered row12073 = true := by decide
set_option maxRecDepth 4096 in
theorem row12073_values : SortedNatTree.values row12073 = Data.row12073 := rfl
theorem row12073_checked : SortedNatTree.contains row12073 = Data.row12073.contains := by
  rw [SortedNatTree.contains_eq_list row12073 row12073_ordered, row12073_values]
def row12097 : SortedNatTree.Tree := (.node 13513 (.node 12109 .empty .empty) (.node 13537 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12097_ordered : SortedNatTree.ordered row12097 = true := by decide
set_option maxRecDepth 4096 in
theorem row12097_values : SortedNatTree.values row12097 = Data.row12097 := rfl
theorem row12097_checked : SortedNatTree.contains row12097 = Data.row12097.contains := by
  rw [SortedNatTree.contains_eq_list row12097 row12097_ordered, row12097_values]
def row12101 : SortedNatTree.Tree := (.node 12437 (.node 12251 .empty .empty) (.node 12689 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12101_ordered : SortedNatTree.ordered row12101 = true := by decide
set_option maxRecDepth 4096 in
theorem row12101_values : SortedNatTree.values row12101 = Data.row12101 := rfl
theorem row12101_checked : SortedNatTree.contains row12101 = Data.row12101.contains := by
  rw [SortedNatTree.contains_eq_list row12101 row12101_ordered, row12101_values]
def row12107 : SortedNatTree.Tree := (.node 12653 (.node 12491 .empty .empty) (.node 13487 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12107_ordered : SortedNatTree.ordered row12107 = true := by decide
set_option maxRecDepth 4096 in
theorem row12107_values : SortedNatTree.values row12107 = Data.row12107 := rfl
theorem row12107_checked : SortedNatTree.contains row12107 = Data.row12107.contains := by
  rw [SortedNatTree.contains_eq_list row12107 row12107_ordered, row12107_values]
def row12109 : SortedNatTree.Tree := (.node 13063 .empty .empty)
set_option maxRecDepth 4096 in
theorem row12109_ordered : SortedNatTree.ordered row12109 = true := by decide
set_option maxRecDepth 4096 in
theorem row12109_values : SortedNatTree.values row12109 = Data.row12109 := rfl
theorem row12109_checked : SortedNatTree.contains row12109 = Data.row12109.contains := by
  rw [SortedNatTree.contains_eq_list row12109 row12109_ordered, row12109_values]
def row12113 : SortedNatTree.Tree := (.node 13457 (.node 12251 .empty .empty) (.node 13679 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12113_ordered : SortedNatTree.ordered row12113 = true := by decide
set_option maxRecDepth 4096 in
theorem row12113_values : SortedNatTree.values row12113 = Data.row12113 := rfl
theorem row12113_checked : SortedNatTree.contains row12113 = Data.row12113.contains := by
  rw [SortedNatTree.contains_eq_list row12113 row12113_ordered, row12113_values]
def row12119 : SortedNatTree.Tree := (.node 13331 (.node 12917 (.node 12743 .empty .empty) .empty) (.node 13463 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12119_ordered : SortedNatTree.ordered row12119 = true := by decide
set_option maxRecDepth 4096 in
theorem row12119_values : SortedNatTree.values row12119 = Data.row12119 := rfl
theorem row12119_checked : SortedNatTree.contains row12119 = Data.row12119.contains := by
  rw [SortedNatTree.contains_eq_list row12119 row12119_ordered, row12119_values]
def row12143 : SortedNatTree.Tree := (.node 12791 (.node 12269 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12143_ordered : SortedNatTree.ordered row12143 = true := by decide
set_option maxRecDepth 4096 in
theorem row12143_values : SortedNatTree.values row12143 = Data.row12143 := rfl
theorem row12143_checked : SortedNatTree.contains row12143 = Data.row12143.contains := by
  rw [SortedNatTree.contains_eq_list row12143 row12143_ordered, row12143_values]
def row12149 : SortedNatTree.Tree := (.node 12821 (.node 12659 .empty .empty) (.node 13829 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12149_ordered : SortedNatTree.ordered row12149 = true := by decide
set_option maxRecDepth 4096 in
theorem row12149_values : SortedNatTree.values row12149 = Data.row12149 := rfl
theorem row12149_checked : SortedNatTree.contains row12149 = Data.row12149.contains := by
  rw [SortedNatTree.contains_eq_list row12149 row12149_ordered, row12149_values]
def row12157 : SortedNatTree.Tree := (.node 13687 (.node 12601 (.node 12241 .empty .empty) .empty) (.node 13711 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12157_ordered : SortedNatTree.ordered row12157 = true := by decide
set_option maxRecDepth 4096 in
theorem row12157_values : SortedNatTree.values row12157 = Data.row12157 := rfl
theorem row12157_checked : SortedNatTree.contains row12157 = Data.row12157.contains := by
  rw [SortedNatTree.contains_eq_list row12157 row12157_ordered, row12157_values]
def row12161 : SortedNatTree.Tree := (.node 13163 (.node 13121 (.node 13001 .empty .empty) .empty) (.node 13337 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12161_ordered : SortedNatTree.ordered row12161 = true := by decide
set_option maxRecDepth 4096 in
theorem row12161_values : SortedNatTree.values row12161 = Data.row12161 := rfl
theorem row12161_checked : SortedNatTree.contains row12161 = Data.row12161.contains := by
  rw [SortedNatTree.contains_eq_list row12161 row12161_ordered, row12161_values]
def row12163 : SortedNatTree.Tree := (.node 13171 (.node 12739 (.node 12457 .empty .empty) .empty) (.node 13591 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12163_ordered : SortedNatTree.ordered row12163 = true := by decide
set_option maxRecDepth 4096 in
theorem row12163_values : SortedNatTree.values row12163 = Data.row12163 := rfl
theorem row12163_checked : SortedNatTree.contains row12163 = Data.row12163.contains := by
  rw [SortedNatTree.contains_eq_list row12163 row12163_ordered, row12163_values]
def row12197 : SortedNatTree.Tree := (.node 13367 (.node 12569 .empty .empty) (.node 13499 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12197_ordered : SortedNatTree.ordered row12197 = true := by decide
set_option maxRecDepth 4096 in
theorem row12197_values : SortedNatTree.values row12197 = Data.row12197 := rfl
theorem row12197_checked : SortedNatTree.contains row12197 = Data.row12197.contains := by
  rw [SortedNatTree.contains_eq_list row12197 row12197_ordered, row12197_values]
def row12203 : SortedNatTree.Tree := (.node 13421 .empty .empty)
set_option maxRecDepth 4096 in
theorem row12203_ordered : SortedNatTree.ordered row12203 = true := by decide
set_option maxRecDepth 4096 in
theorem row12203_values : SortedNatTree.values row12203 = Data.row12203 := rfl
theorem row12203_checked : SortedNatTree.contains row12203 = Data.row12203.contains := by
  rw [SortedNatTree.contains_eq_list row12203 row12203_ordered, row12203_values]
def row12211 : SortedNatTree.Tree := (.node 13807 (.node 13633 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12211_ordered : SortedNatTree.ordered row12211 = true := by decide
set_option maxRecDepth 4096 in
theorem row12211_values : SortedNatTree.values row12211 = Data.row12211 := rfl
theorem row12211_checked : SortedNatTree.contains row12211 = Data.row12211.contains := by
  rw [SortedNatTree.contains_eq_list row12211 row12211_ordered, row12211_values]
def row12227 : SortedNatTree.Tree := (.node 12671 (.node 12479 .empty .empty) (.node 13001 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12227_ordered : SortedNatTree.ordered row12227 = true := by decide
set_option maxRecDepth 4096 in
theorem row12227_values : SortedNatTree.values row12227 = Data.row12227 := rfl
theorem row12227_checked : SortedNatTree.contains row12227 = Data.row12227.contains := by
  rw [SortedNatTree.contains_eq_list row12227 row12227_ordered, row12227_values]
def row12239 : SortedNatTree.Tree := (.node 13463 .empty .empty)
set_option maxRecDepth 4096 in
theorem row12239_ordered : SortedNatTree.ordered row12239 = true := by decide
set_option maxRecDepth 4096 in
theorem row12239_values : SortedNatTree.values row12239 = Data.row12239 := rfl
theorem row12239_checked : SortedNatTree.contains row12239 = Data.row12239.contains := by
  rw [SortedNatTree.contains_eq_list row12239 row12239_ordered, row12239_values]
def row12241 : SortedNatTree.Tree := (.node 13591 (.node 13003 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12241_ordered : SortedNatTree.ordered row12241 = true := by decide
set_option maxRecDepth 4096 in
theorem row12241_values : SortedNatTree.values row12241 = Data.row12241 := rfl
theorem row12241_checked : SortedNatTree.contains row12241 = Data.row12241.contains := by
  rw [SortedNatTree.contains_eq_list row12241 row12241_ordered, row12241_values]
def row12251 : SortedNatTree.Tree := (.node 13151 (.node 13037 (.node 12893 .empty .empty) .empty) (.node 13187 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12251_ordered : SortedNatTree.ordered row12251 = true := by decide
set_option maxRecDepth 4096 in
theorem row12251_values : SortedNatTree.values row12251 = Data.row12251 := rfl
theorem row12251_checked : SortedNatTree.contains row12251 = Data.row12251.contains := by
  rw [SortedNatTree.contains_eq_list row12251 row12251_ordered, row12251_values]
def row12253 : SortedNatTree.Tree := (.node 13183 (.node 12721 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12253_ordered : SortedNatTree.ordered row12253 = true := by decide
set_option maxRecDepth 4096 in
theorem row12253_values : SortedNatTree.values row12253 = Data.row12253 := rfl
theorem row12253_checked : SortedNatTree.contains row12253 = Data.row12253.contains := by
  rw [SortedNatTree.contains_eq_list row12253 row12253_ordered, row12253_values]
def row12263 : SortedNatTree.Tree := (.node 13721 (.node 12917 .empty .empty) (.node 13763 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12263_ordered : SortedNatTree.ordered row12263 = true := by decide
set_option maxRecDepth 4096 in
theorem row12263_values : SortedNatTree.values row12263 = Data.row12263 := rfl
theorem row12263_checked : SortedNatTree.contains row12263 = Data.row12263.contains := by
  rw [SortedNatTree.contains_eq_list row12263 row12263_ordered, row12263_values]
def row12269 : SortedNatTree.Tree := (.node 12671 .empty .empty)
set_option maxRecDepth 4096 in
theorem row12269_ordered : SortedNatTree.ordered row12269 = true := by decide
set_option maxRecDepth 4096 in
theorem row12269_values : SortedNatTree.values row12269 = Data.row12269 := rfl
theorem row12269_checked : SortedNatTree.contains row12269 = Data.row12269.contains := by
  rw [SortedNatTree.contains_eq_list row12269 row12269_ordered, row12269_values]
def row12277 : SortedNatTree.Tree := (.node 12637 (.node 12373 .empty .empty) (.node 13411 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12277_ordered : SortedNatTree.ordered row12277 = true := by decide
set_option maxRecDepth 4096 in
theorem row12277_values : SortedNatTree.values row12277 = Data.row12277 := rfl
theorem row12277_checked : SortedNatTree.contains row12277 = Data.row12277.contains := by
  rw [SortedNatTree.contains_eq_list row12277 row12277_ordered, row12277_values]
def row12281 : SortedNatTree.Tree := (.node 13499 (.node 13121 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12281_ordered : SortedNatTree.ordered row12281 = true := by decide
set_option maxRecDepth 4096 in
theorem row12281_values : SortedNatTree.values row12281 = Data.row12281 := rfl
theorem row12281_checked : SortedNatTree.contains row12281 = Data.row12281.contains := by
  rw [SortedNatTree.contains_eq_list row12281 row12281_ordered, row12281_values]
def row12289 : SortedNatTree.Tree := .empty
set_option maxRecDepth 4096 in
theorem row12289_ordered : SortedNatTree.ordered row12289 = true := by decide
set_option maxRecDepth 4096 in
theorem row12289_values : SortedNatTree.values row12289 = Data.row12289 := rfl
theorem row12289_checked : SortedNatTree.contains row12289 = Data.row12289.contains := by
  rw [SortedNatTree.contains_eq_list row12289 row12289_ordered, row12289_values]
def row12301 : SortedNatTree.Tree := (.node 13249 (.node 12967 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12301_ordered : SortedNatTree.ordered row12301 = true := by decide
set_option maxRecDepth 4096 in
theorem row12301_values : SortedNatTree.values row12301 = Data.row12301 := rfl
theorem row12301_checked : SortedNatTree.contains row12301 = Data.row12301.contains := by
  rw [SortedNatTree.contains_eq_list row12301 row12301_ordered, row12301_values]
def row12323 : SortedNatTree.Tree := (.node 13259 (.node 12953 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row12323_ordered : SortedNatTree.ordered row12323 = true := by decide
set_option maxRecDepth 4096 in
theorem row12323_values : SortedNatTree.values row12323 = Data.row12323 := rfl
theorem row12323_checked : SortedNatTree.contains row12323 = Data.row12323.contains := by
  rw [SortedNatTree.contains_eq_list row12323 row12323_ordered, row12323_values]
def row12329 : SortedNatTree.Tree := (.node 12893 .empty .empty)
set_option maxRecDepth 4096 in
theorem row12329_ordered : SortedNatTree.ordered row12329 = true := by decide
set_option maxRecDepth 4096 in
theorem row12329_values : SortedNatTree.values row12329 = Data.row12329 := rfl
theorem row12329_checked : SortedNatTree.contains row12329 = Data.row12329.contains := by
  rw [SortedNatTree.contains_eq_list row12329 row12329_ordered, row12329_values]
def row12343 : SortedNatTree.Tree := (.node 13513 (.node 13477 (.node 12697 .empty .empty) .empty) (.node 13669 .empty .empty))
set_option maxRecDepth 4096 in
theorem row12343_ordered : SortedNatTree.ordered row12343 = true := by decide
set_option maxRecDepth 4096 in
theorem row12343_values : SortedNatTree.values row12343 = Data.row12343 := rfl
theorem row12343_checked : SortedNatTree.contains row12343 = Data.row12343.contains := by
  rw [SortedNatTree.contains_eq_list row12343 row12343_ordered, row12343_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
