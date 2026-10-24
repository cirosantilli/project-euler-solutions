import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row8747 : SortedNatTree.Tree := (.node 10853 (.node 9851 (.node 9479 (.node 9377 .empty .empty) .empty) (.node 10589 (.node 10139 .empty .empty) .empty)) (.node 14033 (.node 12401 (.node 11351 .empty .empty) .empty) (.node 16529 (.node 15227 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8747_ordered : SortedNatTree.ordered row8747 = true := by decide
set_option maxRecDepth 4096 in
theorem row8747_values : SortedNatTree.values row8747 = Data.row8747 := rfl
theorem row8747_checked : SortedNatTree.contains row8747 = Data.row8747.contains := by
  rw [SortedNatTree.contains_eq_list row8747 row8747_ordered, row8747_values]
def row8753 : SortedNatTree.Tree := (.node 12203 (.node 11279 (.node 9041 (.node 8819 .empty .empty) (.node 9533 .empty .empty)) (.node 11903 (.node 11777 .empty .empty) .empty)) (.node 15803 (.node 14639 (.node 14081 .empty .empty) (.node 14723 .empty .empty)) (.node 17021 (.node 16223 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8753_ordered : SortedNatTree.ordered row8753 = true := by decide
set_option maxRecDepth 4096 in
theorem row8753_values : SortedNatTree.values row8753 = Data.row8753 := rfl
theorem row8753_checked : SortedNatTree.contains row8753 = Data.row8753.contains := by
  rw [SortedNatTree.contains_eq_list row8753 row8753_ordered, row8753_values]
def row8761 : SortedNatTree.Tree := (.node 12373 (.node 10267 (.node 9619 (.node 9613 .empty .empty) (.node 9787 .empty .empty)) (.node 10789 (.node 10321 .empty .empty) (.node 10993 .empty .empty))) (.node 14887 (.node 13669 (.node 12583 .empty .empty) (.node 14143 .empty .empty)) (.node 16453 (.node 15619 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8761_ordered : SortedNatTree.ordered row8761 = true := by decide
set_option maxRecDepth 4096 in
theorem row8761_values : SortedNatTree.values row8761 = Data.row8761 := rfl
theorem row8761_checked : SortedNatTree.contains row8761 = Data.row8761.contains := by
  rw [SortedNatTree.contains_eq_list row8761 row8761_ordered, row8761_values]
def row8779 : SortedNatTree.Tree := (.node 13693 (.node 12823 (.node 11071 (.node 9697 .empty .empty) .empty) (.node 13567 (.node 13063 .empty .empty) .empty)) (.node 16759 (.node 16339 (.node 15823 .empty .empty) .empty) (.node 17041 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8779_ordered : SortedNatTree.ordered row8779 = true := by decide
set_option maxRecDepth 4096 in
theorem row8779_values : SortedNatTree.values row8779 = Data.row8779 := rfl
theorem row8779_checked : SortedNatTree.contains row8779 = Data.row8779.contains := by
  rw [SortedNatTree.contains_eq_list row8779 row8779_ordered, row8779_values]
def row8783 : SortedNatTree.Tree := (.node 12143 (.node 9767 (.node 9473 (.node 9281 .empty .empty) (.node 9551 .empty .empty)) (.node 11657 (.node 9803 .empty .empty) .empty)) (.node 15077 (.node 13463 (.node 12269 .empty .empty) (.node 14699 .empty .empty)) (.node 17207 (.node 16427 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8783_ordered : SortedNatTree.ordered row8783 = true := by decide
set_option maxRecDepth 4096 in
theorem row8783_values : SortedNatTree.values row8783 = Data.row8783 := rfl
theorem row8783_checked : SortedNatTree.contains row8783 = Data.row8783.contains := by
  rw [SortedNatTree.contains_eq_list row8783 row8783_ordered, row8783_values]
def row8803 : SortedNatTree.Tree := (.node 12967 (.node 11827 (.node 10531 (.node 9871 .empty .empty) (.node 11491 .empty .empty)) (.node 12409 (.node 12049 .empty .empty) .empty)) (.node 15451 (.node 14947 (.node 13933 .empty .empty) .empty) (.node 16747 (.node 16693 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8803_ordered : SortedNatTree.ordered row8803 = true := by decide
set_option maxRecDepth 4096 in
theorem row8803_values : SortedNatTree.values row8803 = Data.row8803 := rfl
theorem row8803_checked : SortedNatTree.contains row8803 = Data.row8803.contains := by
  rw [SortedNatTree.contains_eq_list row8803 row8803_ordered, row8803_values]
def row8807 : SortedNatTree.Tree := (.node 11801 (.node 10139 (.node 9239 (.node 9209 .empty .empty) .empty) (.node 11369 (.node 10301 .empty .empty) .empty)) (.node 16787 (.node 13901 (.node 11927 .empty .empty) .empty) (.node 16871 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8807_ordered : SortedNatTree.ordered row8807 = true := by decide
set_option maxRecDepth 4096 in
theorem row8807_values : SortedNatTree.values row8807 = Data.row8807 := rfl
theorem row8807_checked : SortedNatTree.contains row8807 = Data.row8807.contains := by
  rw [SortedNatTree.contains_eq_list row8807 row8807_ordered, row8807_values]
def row8819 : SortedNatTree.Tree := (.node 13931 (.node 11171 (.node 9719 (.node 9239 .empty .empty) (.node 10181 .empty .empty)) (.node 13829 (.node 11411 .empty .empty) .empty)) (.node 15383 (.node 14057 (.node 13997 .empty .empty) .empty) (.node 17123 (.node 16619 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8819_ordered : SortedNatTree.ordered row8819 = true := by decide
set_option maxRecDepth 4096 in
theorem row8819_values : SortedNatTree.values row8819 = Data.row8819 := rfl
theorem row8819_checked : SortedNatTree.contains row8819 = Data.row8819.contains := by
  rw [SortedNatTree.contains_eq_list row8819 row8819_ordered, row8819_values]
def row8821 : SortedNatTree.Tree := (.node 12043 (.node 10459 (.node 9319 (.node 9067 .empty .empty) (.node 9883 .empty .empty)) (.node 11173 (.node 10861 .empty .empty) .empty)) (.node 15319 (.node 14281 (.node 13381 .empty .empty) .empty) (.node 16561 (.node 15607 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8821_ordered : SortedNatTree.ordered row8821 = true := by decide
set_option maxRecDepth 4096 in
theorem row8821_values : SortedNatTree.values row8821 = Data.row8821 := rfl
theorem row8821_checked : SortedNatTree.contains row8821 = Data.row8821.contains := by
  rw [SortedNatTree.contains_eq_list row8821 row8821_ordered, row8821_values]
def row8831 : SortedNatTree.Tree := (.node 12203 (.node 10301 (.node 9293 (.node 8837 .empty .empty) .empty) (.node 11399 .empty .empty)) (.node 14009 (.node 13109 .empty .empty) (.node 16061 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8831_ordered : SortedNatTree.ordered row8831 = true := by decide
set_option maxRecDepth 4096 in
theorem row8831_values : SortedNatTree.values row8831 = Data.row8831 := rfl
theorem row8831_checked : SortedNatTree.contains row8831 = Data.row8831.contains := by
  rw [SortedNatTree.contains_eq_list row8831 row8831_ordered, row8831_values]
def row8837 : SortedNatTree.Tree := (.node 11657 (.node 10601 (.node 9371 (.node 8951 .empty .empty) (.node 9623 .empty .empty)) (.node 11483 (.node 11399 .empty .empty) .empty)) (.node 16187 (.node 12491 (.node 12197 .empty .empty) (.node 13463 .empty .empty)) (.node 17117 (.node 16811 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8837_ordered : SortedNatTree.ordered row8837 = true := by decide
set_option maxRecDepth 4096 in
theorem row8837_values : SortedNatTree.values row8837 = Data.row8837 := rfl
theorem row8837_checked : SortedNatTree.contains row8837 = Data.row8837.contains := by
  rw [SortedNatTree.contains_eq_list row8837 row8837_ordered, row8837_values]
def row8839 : SortedNatTree.Tree := (.node 12979 (.node 11971 (.node 11047 (.node 10429 .empty .empty) .empty) (.node 12829 .empty .empty)) (.node 14929 (.node 14653 (.node 13441 .empty .empty) .empty) (.node 16603 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8839_ordered : SortedNatTree.ordered row8839 = true := by decide
set_option maxRecDepth 4096 in
theorem row8839_values : SortedNatTree.values row8839 = Data.row8839 := rfl
theorem row8839_checked : SortedNatTree.contains row8839 = Data.row8839.contains := by
  rw [SortedNatTree.contains_eq_list row8839 row8839_ordered, row8839_values]
def row8849 : SortedNatTree.Tree := (.node 11939 (.node 10193 (.node 9419 (.node 9041 .empty .empty) (.node 10079 .empty .empty)) (.node 11789 (.node 10781 .empty .empty) .empty)) (.node 14633 (.node 13331 (.node 12653 .empty .empty) (.node 14423 .empty .empty)) (.node 16073 (.node 15497 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8849_ordered : SortedNatTree.ordered row8849 = true := by decide
set_option maxRecDepth 4096 in
theorem row8849_values : SortedNatTree.values row8849 = Data.row8849 := rfl
theorem row8849_checked : SortedNatTree.contains row8849 = Data.row8849.contains := by
  rw [SortedNatTree.contains_eq_list row8849 row8849_ordered, row8849_values]
def row8861 : SortedNatTree.Tree := (.node 11969 (.node 10343 (.node 9887 (.node 9029 .empty .empty) (.node 9941 .empty .empty)) (.node 11549 (.node 10781 .empty .empty) (.node 11939 .empty .empty))) (.node 14087 (.node 13619 (.node 12011 .empty .empty) (.node 13649 .empty .empty)) (.node 15767 (.node 14753 .empty .empty) (.node 16229 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8861_ordered : SortedNatTree.ordered row8861 = true := by decide
set_option maxRecDepth 4096 in
theorem row8861_values : SortedNatTree.values row8861 = Data.row8861 := rfl
theorem row8861_checked : SortedNatTree.contains row8861 = Data.row8861.contains := by
  rw [SortedNatTree.contains_eq_list row8861 row8861_ordered, row8861_values]
def row8863 : SortedNatTree.Tree := (.node 11827 (.node 9871 (.node 9277 (.node 8887 .empty .empty) (.node 9349 .empty .empty)) (.node 11131 (.node 10753 .empty .empty) (.node 11299 .empty .empty))) (.node 14629 (.node 13177 (.node 12919 .empty .empty) (.node 14437 .empty .empty)) (.node 15277 (.node 14851 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8863_ordered : SortedNatTree.ordered row8863 = true := by decide
set_option maxRecDepth 4096 in
theorem row8863_values : SortedNatTree.values row8863 = Data.row8863 := rfl
theorem row8863_checked : SortedNatTree.contains row8863 = Data.row8863.contains := by
  rw [SortedNatTree.contains_eq_list row8863 row8863_ordered, row8863_values]
def row8867 : SortedNatTree.Tree := (.node 13007 (.node 11927 (.node 10247 (.node 9323 .empty .empty) (.node 11909 .empty .empty)) (.node 12923 (.node 12269 .empty .empty) .empty)) (.node 16427 (.node 15737 (.node 13649 .empty .empty) .empty) (.node 16703 (.node 16661 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8867_ordered : SortedNatTree.ordered row8867 = true := by decide
set_option maxRecDepth 4096 in
theorem row8867_values : SortedNatTree.values row8867 = Data.row8867 := rfl
theorem row8867_checked : SortedNatTree.contains row8867 = Data.row8867.contains := by
  rw [SortedNatTree.contains_eq_list row8867 row8867_ordered, row8867_values]
def row8887 : SortedNatTree.Tree := (.node 13183 (.node 12301 (.node 12073 (.node 9601 .empty .empty) .empty) (.node 13093 (.node 12637 .empty .empty) .empty)) (.node 15031 (.node 14149 (.node 13219 .empty .empty) .empty) (.node 15859 (.node 15277 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8887_ordered : SortedNatTree.ordered row8887 = true := by decide
set_option maxRecDepth 4096 in
theorem row8887_values : SortedNatTree.values row8887 = Data.row8887 := rfl
theorem row8887_checked : SortedNatTree.contains row8887 = Data.row8887.contains := by
  rw [SortedNatTree.contains_eq_list row8887 row8887_ordered, row8887_values]
def row8893 : SortedNatTree.Tree := (.node 12577 (.node 9973 (.node 9421 (.node 9187 .empty .empty) (.node 9649 .empty .empty)) (.node 12037 (.node 11953 .empty .empty) (.node 12301 .empty .empty))) (.node 15727 (.node 14011 (.node 13627 .empty .empty) (.node 14731 .empty .empty)) (.node 16069 (.node 16063 .empty .empty) (.node 17041 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8893_ordered : SortedNatTree.ordered row8893 = true := by decide
set_option maxRecDepth 4096 in
theorem row8893_values : SortedNatTree.values row8893 = Data.row8893 := rfl
theorem row8893_checked : SortedNatTree.contains row8893 = Data.row8893.contains := by
  rw [SortedNatTree.contains_eq_list row8893 row8893_ordered, row8893_values]
def row8923 : SortedNatTree.Tree := (.node 10711 (.node 9721 (.node 9091 (.node 9043 .empty .empty) (.node 9343 .empty .empty)) (.node 10099 (.node 9883 .empty .empty) .empty)) (.node 13183 (.node 12601 (.node 11593 .empty .empty) .empty) (.node 15973 (.node 14083 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8923_ordered : SortedNatTree.ordered row8923 = true := by decide
set_option maxRecDepth 4096 in
theorem row8923_values : SortedNatTree.values row8923 = Data.row8923 := rfl
theorem row8923_checked : SortedNatTree.contains row8923 = Data.row8923.contains := by
  rw [SortedNatTree.contains_eq_list row8923 row8923_ordered, row8923_values]
def row8929 : SortedNatTree.Tree := (.node 11527 (.node 10069 (.node 9733 (.node 9319 .empty .empty) (.node 9817 .empty .empty)) (.node 10861 (.node 10141 .empty .empty) (.node 10987 .empty .empty))) (.node 14983 (.node 12577 (.node 11863 .empty .empty) (.node 14887 .empty .empty)) (.node 16417 (.node 15031 .empty .empty) (.node 16903 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8929_ordered : SortedNatTree.ordered row8929 = true := by decide
set_option maxRecDepth 4096 in
theorem row8929_values : SortedNatTree.values row8929 = Data.row8929 := rfl
theorem row8929_checked : SortedNatTree.contains row8929 = Data.row8929.contains := by
  rw [SortedNatTree.contains_eq_list row8929 row8929_ordered, row8929_values]
def row8933 : SortedNatTree.Tree := (.node 13103 (.node 10739 (.node 9461 (.node 9377 .empty .empty) (.node 9491 .empty .empty)) (.node 12071 (.node 11213 .empty .empty) .empty)) (.node 15581 (.node 14951 (.node 13451 .empty .empty) (.node 15233 .empty .empty)) (.node 16421 (.node 15767 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8933_ordered : SortedNatTree.ordered row8933 = true := by decide
set_option maxRecDepth 4096 in
theorem row8933_values : SortedNatTree.values row8933 = Data.row8933 := rfl
theorem row8933_checked : SortedNatTree.contains row8933 = Data.row8933.contains := by
  rw [SortedNatTree.contains_eq_list row8933 row8933_ordered, row8933_values]
def row8941 : SortedNatTree.Tree := (.node 14869 (.node 14449 (.node 10831 (.node 9067 .empty .empty) .empty) (.node 14629 .empty .empty)) (.node 16417 (.node 15973 (.node 15307 .empty .empty) .empty) (.node 16567 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8941_ordered : SortedNatTree.ordered row8941 = true := by decide
set_option maxRecDepth 4096 in
theorem row8941_values : SortedNatTree.values row8941 = Data.row8941 := rfl
theorem row8941_checked : SortedNatTree.contains row8941 = Data.row8941.contains := by
  rw [SortedNatTree.contains_eq_list row8941 row8941_ordered, row8941_values]
def row8951 : SortedNatTree.Tree := (.node 13001 (.node 10949 (.node 10499 (.node 10193 .empty .empty) (.node 10631 .empty .empty)) (.node 12143 (.node 11423 .empty .empty) (.node 12809 .empty .empty))) (.node 14321 (.node 13859 (.node 13679 .empty .empty) (.node 14177 .empty .empty)) (.node 15101 (.node 14327 .empty .empty) (.node 16301 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8951_ordered : SortedNatTree.ordered row8951 = true := by decide
set_option maxRecDepth 4096 in
theorem row8951_values : SortedNatTree.values row8951 = Data.row8951 := rfl
theorem row8951_checked : SortedNatTree.contains row8951 = Data.row8951.contains := by
  rw [SortedNatTree.contains_eq_list row8951 row8951_ordered, row8951_values]
def row8963 : SortedNatTree.Tree := (.node 13421 (.node 11489 (.node 11261 (.node 11093 .empty .empty) (.node 11483 .empty .empty)) (.node 12899 (.node 12281 .empty .empty) .empty)) (.node 14537 (.node 14087 (.node 14033 .empty .empty) .empty) (.node 16901 (.node 15683 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8963_ordered : SortedNatTree.ordered row8963 = true := by decide
set_option maxRecDepth 4096 in
theorem row8963_values : SortedNatTree.values row8963 = Data.row8963 := rfl
theorem row8963_checked : SortedNatTree.contains row8963 = Data.row8963.contains := by
  rw [SortedNatTree.contains_eq_list row8963 row8963_ordered, row8963_values]
def row8969 : SortedNatTree.Tree := (.node 14657 (.node 11489 (.node 9539 (.node 9377 .empty .empty) .empty) (.node 14159 (.node 12959 .empty .empty) .empty)) (.node 15443 (.node 15413 (.node 15233 .empty .empty) .empty) (.node 16421 (.node 16301 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8969_ordered : SortedNatTree.ordered row8969 = true := by decide
set_option maxRecDepth 4096 in
theorem row8969_values : SortedNatTree.values row8969 = Data.row8969 := rfl
theorem row8969_checked : SortedNatTree.contains row8969 = Data.row8969.contains := by
  rw [SortedNatTree.contains_eq_list row8969 row8969_ordered, row8969_values]
def row8971 : SortedNatTree.Tree := (.node 12973 (.node 11497 (.node 9769 (.node 9739 (.node 9187 .empty .empty) .empty) (.node 9787 .empty .empty)) (.node 12301 (.node 11833 (.node 11779 .empty .empty) .empty) (.node 12889 .empty .empty))) (.node 15787 (.node 15277 (.node 14341 (.node 13297 .empty .empty) .empty) (.node 15607 .empty .empty)) (.node 15937 (.node 15859 .empty .empty) (.node 15991 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8971_ordered : SortedNatTree.ordered row8971 = true := by decide
set_option maxRecDepth 4096 in
theorem row8971_values : SortedNatTree.values row8971 = Data.row8971 := rfl
theorem row8971_checked : SortedNatTree.contains row8971 = Data.row8971.contains := by
  rw [SortedNatTree.contains_eq_list row8971 row8971_ordered, row8971_values]
def row8999 : SortedNatTree.Tree := (.node 11399 (.node 10601 (.node 9521 (.node 9311 .empty .empty) (.node 9857 .empty .empty)) (.node 10937 (.node 10799 .empty .empty) (.node 11273 .empty .empty))) (.node 15551 (.node 13577 (.node 13457 .empty .empty) (.node 14891 .empty .empty)) (.node 16481 (.node 15629 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8999_ordered : SortedNatTree.ordered row8999 = true := by decide
set_option maxRecDepth 4096 in
theorem row8999_values : SortedNatTree.values row8999 = Data.row8999 := rfl
theorem row8999_checked : SortedNatTree.contains row8999 = Data.row8999.contains := by
  rw [SortedNatTree.contains_eq_list row8999 row8999_ordered, row8999_values]
def row9001 : SortedNatTree.Tree := (.node 11353 (.node 9901 (.node 9829 (.node 9049 .empty .empty) (.node 9859 .empty .empty)) (.node 10837 (.node 10723 .empty .empty) (.node 11287 .empty .empty))) (.node 13633 (.node 12841 (.node 12781 .empty .empty) (.node 12853 .empty .empty)) (.node 15313 (.node 14029 .empty .empty) (.node 16831 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9001_ordered : SortedNatTree.ordered row9001 = true := by decide
set_option maxRecDepth 4096 in
theorem row9001_values : SortedNatTree.values row9001 = Data.row9001 := rfl
theorem row9001_checked : SortedNatTree.contains row9001 = Data.row9001.contains := by
  rw [SortedNatTree.contains_eq_list row9001 row9001_ordered, row9001_values]
def row9007 : SortedNatTree.Tree := (.node 14923 (.node 11689 (.node 10009 (.node 9631 .empty .empty) .empty) (.node 13399 .empty .empty)) (.node 15493 (.node 15427 (.node 15073 .empty .empty) .empty) (.node 16759 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9007_ordered : SortedNatTree.ordered row9007 = true := by decide
set_option maxRecDepth 4096 in
theorem row9007_values : SortedNatTree.values row9007 = Data.row9007 := rfl
theorem row9007_checked : SortedNatTree.contains row9007 = Data.row9007.contains := by
  rw [SortedNatTree.contains_eq_list row9007 row9007_ordered, row9007_values]
def row9011 : SortedNatTree.Tree := (.node 12503 (.node 10589 (.node 9311 (.node 9209 .empty .empty) (.node 9749 .empty .empty)) (.node 11657 (.node 11087 .empty .empty) .empty)) (.node 15377 (.node 15359 (.node 14561 .empty .empty) .empty) (.node 16829 (.node 16673 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9011_ordered : SortedNatTree.ordered row9011 = true := by decide
set_option maxRecDepth 4096 in
theorem row9011_values : SortedNatTree.values row9011 = Data.row9011 := rfl
theorem row9011_checked : SortedNatTree.contains row9011 = Data.row9011.contains := by
  rw [SortedNatTree.contains_eq_list row9011 row9011_ordered, row9011_values]
def row9013 : SortedNatTree.Tree := (.node 13267 (.node 9349 (.node 9241 (.node 9199 .empty .empty) .empty) (.node 13099 .empty .empty)) (.node 15361 (.node 15307 .empty .empty) (.node 16231 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9013_ordered : SortedNatTree.ordered row9013 = true := by decide
set_option maxRecDepth 4096 in
theorem row9013_values : SortedNatTree.values row9013 = Data.row9013 := rfl
theorem row9013_checked : SortedNatTree.contains row9013 = Data.row9013.contains := by
  rw [SortedNatTree.contains_eq_list row9013 row9013_ordered, row9013_values]
def row9029 : SortedNatTree.Tree := (.node 14249 (.node 10391 (.node 9491 (.node 9281 .empty .empty) (.node 9743 .empty .empty)) (.node 11927 (.node 10847 .empty .empty) (.node 12011 .empty .empty))) (.node 15737 (.node 15467 (.node 14717 .empty .empty) (.node 15581 .empty .empty)) (.node 16829 (.node 16691 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9029_ordered : SortedNatTree.ordered row9029 = true := by decide
set_option maxRecDepth 4096 in
theorem row9029_values : SortedNatTree.values row9029 = Data.row9029 := rfl
theorem row9029_checked : SortedNatTree.contains row9029 = Data.row9029.contains := by
  rw [SortedNatTree.contains_eq_list row9029 row9029_ordered, row9029_values]
def row9041 : SortedNatTree.Tree := (.node 11027 (.node 9851 (.node 9521 (.node 9437 .empty .empty) (.node 9551 .empty .empty)) (.node 10301 (.node 10223 .empty .empty) .empty)) (.node 12203 (.node 11093 (.node 11069 .empty .empty) (.node 11987 .empty .empty)) (.node 15053 (.node 12239 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9041_ordered : SortedNatTree.ordered row9041 = true := by decide
set_option maxRecDepth 4096 in
theorem row9041_values : SortedNatTree.values row9041 = Data.row9041 := rfl
theorem row9041_checked : SortedNatTree.contains row9041 = Data.row9041.contains := by
  rw [SortedNatTree.contains_eq_list row9041 row9041_ordered, row9041_values]
def row9043 : SortedNatTree.Tree := (.node 13381 (.node 11491 (.node 10093 (.node 9157 .empty .empty) .empty) (.node 13093 (.node 12037 .empty .empty) .empty)) (.node 14533 (.node 14347 (.node 13537 .empty .empty) .empty) (.node 16561 (.node 15991 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9043_ordered : SortedNatTree.ordered row9043 = true := by decide
set_option maxRecDepth 4096 in
theorem row9043_values : SortedNatTree.values row9043 = Data.row9043 := rfl
theorem row9043_checked : SortedNatTree.contains row9043 = Data.row9043.contains := by
  rw [SortedNatTree.contains_eq_list row9043 row9043_ordered, row9043_values]
def row9049 : SortedNatTree.Tree := (.node 11863 (.node 11047 (.node 9973 (.node 9613 (.node 9187 .empty .empty) .empty) (.node 10627 .empty .empty)) (.node 11383 (.node 11311 (.node 11083 .empty .empty) .empty) (.node 11551 .empty .empty))) (.node 14713 (.node 12637 (.node 12451 (.node 12097 .empty .empty) .empty) (.node 14533 .empty .empty)) (.node 16417 (.node 16231 (.node 15241 .empty .empty) .empty) (.node 16633 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9049_ordered : SortedNatTree.ordered row9049 = true := by decide
set_option maxRecDepth 4096 in
theorem row9049_values : SortedNatTree.values row9049 = Data.row9049 := rfl
theorem row9049_checked : SortedNatTree.contains row9049 = Data.row9049.contains := by
  rw [SortedNatTree.contains_eq_list row9049 row9049_ordered, row9049_values]
def row9059 : SortedNatTree.Tree := (.node 13691 (.node 10781 (.node 10151 (.node 9473 .empty .empty) (.node 10391 .empty .empty)) (.node 11777 (.node 11411 .empty .empty) (.node 12653 .empty .empty))) (.node 15401 (.node 13913 (.node 13751 .empty .empty) (.node 14627 .empty .empty)) (.node 16139 (.node 15749 .empty .empty) (.node 16631 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9059_ordered : SortedNatTree.ordered row9059 = true := by decide
set_option maxRecDepth 4096 in
theorem row9059_values : SortedNatTree.values row9059 = Data.row9059 := rfl
theorem row9059_checked : SortedNatTree.contains row9059 = Data.row9059.contains := by
  rw [SortedNatTree.contains_eq_list row9059 row9059_ordered, row9059_values]
def row9067 : SortedNatTree.Tree := (.node 13477 (.node 10501 (.node 9811 (.node 9769 (.node 9349 .empty .empty) .empty) (.node 10369 .empty .empty)) (.node 10891 (.node 10687 .empty .empty) (.node 11551 .empty .empty))) (.node 15289 (.node 14173 (.node 13627 .empty .empty) (.node 14503 .empty .empty)) (.node 15787 (.node 15619 .empty .empty) (.node 16249 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9067_ordered : SortedNatTree.ordered row9067 = true := by decide
set_option maxRecDepth 4096 in
theorem row9067_values : SortedNatTree.values row9067 = Data.row9067 := rfl
theorem row9067_checked : SortedNatTree.contains row9067 = Data.row9067.contains := by
  rw [SortedNatTree.contains_eq_list row9067 row9067_ordered, row9067_values]
def row9091 : SortedNatTree.Tree := (.node 11299 (.node 10597 (.node 9781 (.node 9733 (.node 9511 .empty .empty) .empty) (.node 10333 .empty .empty)) (.node 10957 (.node 10639 .empty .empty) (.node 11251 .empty .empty))) (.node 14737 (.node 12979 (.node 12907 (.node 12433 .empty .empty) .empty) (.node 13963 .empty .empty)) (.node 16333 (.node 16189 .empty .empty) (.node 16339 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9091_ordered : SortedNatTree.ordered row9091 = true := by decide
set_option maxRecDepth 4096 in
theorem row9091_values : SortedNatTree.values row9091 = Data.row9091 := rfl
theorem row9091_checked : SortedNatTree.contains row9091 = Data.row9091.contains := by
  rw [SortedNatTree.contains_eq_list row9091 row9091_ordered, row9091_values]
def row9103 : SortedNatTree.Tree := (.node 11317 (.node 10369 (.node 9649 (.node 9181 .empty .empty) (.node 10069 .empty .empty)) (.node 10909 (.node 10651 .empty .empty) .empty)) (.node 14983 (.node 12541 (.node 11953 .empty .empty) (.node 14653 .empty .empty)) (.node 15733 (.node 15277 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9103_ordered : SortedNatTree.ordered row9103 = true := by decide
set_option maxRecDepth 4096 in
theorem row9103_values : SortedNatTree.values row9103 = Data.row9103 := rfl
theorem row9103_checked : SortedNatTree.contains row9103 = Data.row9103.contains := by
  rw [SortedNatTree.contains_eq_list row9103 row9103_ordered, row9103_values]
def row9109 : SortedNatTree.Tree := (.node 12553 (.node 9949 (.node 9181 (.node 9133 .empty .empty) (.node 9871 .empty .empty)) (.node 10771 (.node 10723 .empty .empty) (.node 11467 .empty .empty))) (.node 15919 (.node 15121 (.node 15073 .empty .empty) (.node 15349 .empty .empty)) (.node 16381 (.node 16249 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9109_ordered : SortedNatTree.ordered row9109 = true := by decide
set_option maxRecDepth 4096 in
theorem row9109_values : SortedNatTree.values row9109 = Data.row9109 := rfl
theorem row9109_checked : SortedNatTree.contains row9109 = Data.row9109.contains := by
  rw [SortedNatTree.contains_eq_list row9109 row9109_ordered, row9109_values]
def row9127 : SortedNatTree.Tree := (.node 11677 (.node 9679 (.node 9343 (.node 9187 .empty .empty) (.node 9547 .empty .empty)) (.node 10597 (.node 10273 .empty .empty) (.node 10891 .empty .empty))) (.node 14389 (.node 12613 (.node 12379 .empty .empty) (.node 13933 .empty .empty)) (.node 15823 (.node 15511 .empty .empty) (.node 16603 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9127_ordered : SortedNatTree.ordered row9127 = true := by decide
set_option maxRecDepth 4096 in
theorem row9127_values : SortedNatTree.values row9127 = Data.row9127 := rfl
theorem row9127_checked : SortedNatTree.contains row9127 = Data.row9127.contains := by
  rw [SortedNatTree.contains_eq_list row9127 row9127_ordered, row9127_values]
def row9133 : SortedNatTree.Tree := (.node 11353 (.node 9901 (.node 9337 (.node 9319 .empty .empty) (.node 9397 .empty .empty)) (.node 10429 (.node 10039 .empty .empty) .empty)) (.node 14323 (.node 12619 (.node 11527 .empty .empty) (.node 13441 .empty .empty)) (.node 16333 (.node 15889 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9133_ordered : SortedNatTree.ordered row9133 = true := by decide
set_option maxRecDepth 4096 in
theorem row9133_values : SortedNatTree.values row9133 = Data.row9133 := rfl
theorem row9133_checked : SortedNatTree.contains row9133 = Data.row9133.contains := by
  rw [SortedNatTree.contains_eq_list row9133 row9133_ordered, row9133_values]
def row9137 : SortedNatTree.Tree := (.node 12791 (.node 10799 (.node 9689 (.node 9293 .empty .empty) (.node 10331 .empty .empty)) (.node 12377 (.node 11243 .empty .empty) (.node 12479 .empty .empty))) (.node 15017 (.node 13001 (.node 12959 .empty .empty) (.node 14033 .empty .empty)) (.node 16547 (.node 15329 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9137_ordered : SortedNatTree.ordered row9137 = true := by decide
set_option maxRecDepth 4096 in
theorem row9137_values : SortedNatTree.values row9137 = Data.row9137 := rfl
theorem row9137_checked : SortedNatTree.contains row9137 = Data.row9137.contains := by
  rw [SortedNatTree.contains_eq_list row9137 row9137_ordered, row9137_values]
def row9151 : SortedNatTree.Tree := (.node 13063 (.node 11311 (.node 9949 (.node 9739 (.node 9643 .empty .empty) .empty) (.node 10789 (.node 10711 .empty .empty) .empty)) (.node 12763 (.node 11863 (.node 11731 .empty .empty) .empty) (.node 12829 .empty .empty))) (.node 15679 (.node 14449 (.node 13921 (.node 13687 .empty .empty) .empty) (.node 15199 .empty .empty)) (.node 16651 (.node 16183 (.node 16087 .empty .empty) .empty) (.node 16759 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9151_ordered : SortedNatTree.ordered row9151 = true := by decide
set_option maxRecDepth 4096 in
theorem row9151_values : SortedNatTree.values row9151 = Data.row9151 := rfl
theorem row9151_checked : SortedNatTree.contains row9151 = Data.row9151.contains := by
  rw [SortedNatTree.contains_eq_list row9151 row9151_ordered, row9151_values]
def row9157 : SortedNatTree.Tree := (.node 13399 (.node 10723 (.node 9907 (.node 9811 .empty .empty) (.node 10009 .empty .empty)) (.node 12373 (.node 12211 .empty .empty) (.node 12757 .empty .empty))) (.node 15679 (.node 14143 (.node 13441 .empty .empty) (.node 14563 .empty .empty)) (.node 16447 (.node 15817 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9157_ordered : SortedNatTree.ordered row9157 = true := by decide
set_option maxRecDepth 4096 in
theorem row9157_values : SortedNatTree.values row9157 = Data.row9157 := rfl
theorem row9157_checked : SortedNatTree.contains row9157 = Data.row9157.contains := by
  rw [SortedNatTree.contains_eq_list row9157 row9157_ordered, row9157_values]
def row9161 : SortedNatTree.Tree := (.node 11633 (.node 10499 (.node 10253 (.node 10211 (.node 9533 .empty .empty) .empty) (.node 10487 (.node 10271 .empty .empty) .empty)) (.node 11213 (.node 11117 (.node 10589 .empty .empty) .empty) (.node 11279 .empty .empty))) (.node 13217 (.node 12101 (.node 11699 (.node 11681 .empty .empty) .empty) (.node 12479 .empty .empty)) (.node 15107 (.node 13721 (.node 13469 .empty .empty) .empty) (.node 16319 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9161_ordered : SortedNatTree.ordered row9161 = true := by decide
set_option maxRecDepth 4096 in
theorem row9161_values : SortedNatTree.values row9161 = Data.row9161 := rfl
theorem row9161_checked : SortedNatTree.contains row9161 = Data.row9161.contains := by
  rw [SortedNatTree.contains_eq_list row9161 row9161_ordered, row9161_values]
def row9173 : SortedNatTree.Tree := (.node 13037 (.node 11657 (.node 9839 (.node 9677 .empty .empty) (.node 10103 .empty .empty)) (.node 11981 (.node 11909 .empty .empty) (.node 12899 .empty .empty))) (.node 14327 (.node 13499 (.node 13043 .empty .empty) (.node 13967 .empty .empty)) (.node 16007 (.node 14753 .empty .empty) (.node 16553 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9173_ordered : SortedNatTree.ordered row9173 = true := by decide
set_option maxRecDepth 4096 in
theorem row9173_values : SortedNatTree.values row9173 = Data.row9173 := rfl
theorem row9173_checked : SortedNatTree.contains row9173 = Data.row9173.contains := by
  rw [SortedNatTree.contains_eq_list row9173 row9173_ordered, row9173_values]
def row9181 : SortedNatTree.Tree := (.node 13177 (.node 11923 (.node 10657 (.node 9769 .empty .empty) (.node 11743 .empty .empty)) (.node 12763 (.node 12619 .empty .empty) .empty)) (.node 15619 (.node 15361 (.node 14401 .empty .empty) .empty) (.node 16741 (.node 16453 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9181_ordered : SortedNatTree.ordered row9181 = true := by decide
set_option maxRecDepth 4096 in
theorem row9181_values : SortedNatTree.values row9181 = Data.row9181 := rfl
theorem row9181_checked : SortedNatTree.contains row9181 = Data.row9181.contains := by
  rw [SortedNatTree.contains_eq_list row9181 row9181_ordered, row9181_values]
def row9187 : SortedNatTree.Tree := (.node 12211 (.node 10357 (.node 9439 (.node 9349 .empty .empty) .empty) (.node 12037 (.node 11239 .empty .empty) .empty)) (.node 14341 (.node 13177 (.node 12343 .empty .empty) .empty) (.node 15643 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9187_ordered : SortedNatTree.ordered row9187 = true := by decide
set_option maxRecDepth 4096 in
theorem row9187_values : SortedNatTree.values row9187 = Data.row9187 := rfl
theorem row9187_checked : SortedNatTree.contains row9187 = Data.row9187.contains := by
  rw [SortedNatTree.contains_eq_list row9187 row9187_ordered, row9187_values]
def row9199 : SortedNatTree.Tree := (.node 12457 (.node 10111 (.node 9907 .empty .empty) (.node 10333 .empty .empty)) (.node 16267 (.node 12589 .empty .empty) (.node 16573 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9199_ordered : SortedNatTree.ordered row9199 = true := by decide
set_option maxRecDepth 4096 in
theorem row9199_values : SortedNatTree.values row9199 = Data.row9199 := rfl
theorem row9199_checked : SortedNatTree.contains row9199 = Data.row9199.contains := by
  rw [SortedNatTree.contains_eq_list row9199 row9199_ordered, row9199_values]
def row9203 : SortedNatTree.Tree := (.node 12251 (.node 10613 (.node 9719 (.node 9413 .empty .empty) .empty) (.node 11699 .empty .empty)) (.node 14489 (.node 12641 (.node 12473 .empty .empty) .empty) (.node 16193 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9203_ordered : SortedNatTree.ordered row9203 = true := by decide
set_option maxRecDepth 4096 in
theorem row9203_values : SortedNatTree.values row9203 = Data.row9203 := rfl
theorem row9203_checked : SortedNatTree.contains row9203 = Data.row9203.contains := by
  rw [SortedNatTree.contains_eq_list row9203 row9203_ordered, row9203_values]
def row9209 : SortedNatTree.Tree := (.node 12743 (.node 11807 (.node 9629 (.node 9431 .empty .empty) .empty) (.node 12437 (.node 12107 .empty .empty) .empty)) (.node 15887 (.node 14537 (.node 14081 .empty .empty) .empty) (.node 16649 (.node 16319 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9209_ordered : SortedNatTree.ordered row9209 = true := by decide
set_option maxRecDepth 4096 in
theorem row9209_values : SortedNatTree.values row9209 = Data.row9209 := rfl
theorem row9209_checked : SortedNatTree.contains row9209 = Data.row9209.contains := by
  rw [SortedNatTree.contains_eq_list row9209 row9209_ordered, row9209_values]
def row9221 : SortedNatTree.Tree := (.node 12809 (.node 10067 (.node 9791 (.node 9377 .empty .empty) (.node 9851 .empty .empty)) (.node 12479 (.node 10613 .empty .empty) .empty)) (.node 14177 (.node 13103 (.node 12923 .empty .empty) (.node 13691 .empty .empty)) (.node 16787 (.node 15737 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9221_ordered : SortedNatTree.ordered row9221 = true := by decide
set_option maxRecDepth 4096 in
theorem row9221_values : SortedNatTree.values row9221 = Data.row9221 := rfl
theorem row9221_checked : SortedNatTree.contains row9221 = Data.row9221.contains := by
  rw [SortedNatTree.contains_eq_list row9221 row9221_ordered, row9221_values]
def row9227 : SortedNatTree.Tree := (.node 11549 (.node 11087 (.node 10247 (.node 9461 (.node 9311 .empty .empty) .empty) (.node 10457 .empty .empty)) (.node 11423 (.node 11351 .empty .empty) (.node 11483 .empty .empty))) (.node 15569 (.node 14657 (.node 12491 (.node 12149 .empty .empty) .empty) (.node 15329 .empty .empty)) (.node 16187 (.node 15767 .empty .empty) (.node 16451 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9227_ordered : SortedNatTree.ordered row9227 = true := by decide
set_option maxRecDepth 4096 in
theorem row9227_values : SortedNatTree.values row9227 = Data.row9227 := rfl
theorem row9227_checked : SortedNatTree.contains row9227 = Data.row9227.contains := by
  rw [SortedNatTree.contains_eq_list row9227 row9227_ordered, row9227_values]
def row9239 : SortedNatTree.Tree := (.node 13499 (.node 10259 (.node 9941 (.node 9803 (.node 9491 .empty .empty) .empty) (.node 10193 .empty .empty)) (.node 11939 (.node 11519 (.node 11117 .empty .empty) .empty) (.node 13121 .empty .empty))) (.node 14633 (.node 14447 (.node 13997 (.node 13577 .empty .empty) .empty) (.node 14627 .empty .empty)) (.node 16421 (.node 15287 (.node 15173 .empty .empty) .empty) (.node 16691 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9239_ordered : SortedNatTree.ordered row9239 = true := by decide
set_option maxRecDepth 4096 in
theorem row9239_values : SortedNatTree.values row9239 = Data.row9239 := rfl
theorem row9239_checked : SortedNatTree.contains row9239 = Data.row9239.contains := by
  rw [SortedNatTree.contains_eq_list row9239 row9239_ordered, row9239_values]
def row9241 : SortedNatTree.Tree := (.node 14449 (.node 11299 (.node 11239 (.node 10987 .empty .empty) .empty) (.node 12799 (.node 12547 .empty .empty) .empty)) (.node 15679 (.node 15583 (.node 14767 .empty .empty) .empty) (.node 16111 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9241_ordered : SortedNatTree.ordered row9241 = true := by decide
set_option maxRecDepth 4096 in
theorem row9241_values : SortedNatTree.values row9241 = Data.row9241 := rfl
theorem row9241_checked : SortedNatTree.contains row9241 = Data.row9241.contains := by
  rw [SortedNatTree.contains_eq_list row9241 row9241_ordered, row9241_values]
def row9257 : SortedNatTree.Tree := (.node 11909 (.node 11261 (.node 9749 (.node 9371 .empty .empty) .empty) (.node 11903 (.node 11519 .empty .empty) .empty)) (.node 14081 (.node 13883 (.node 13163 .empty .empty) .empty) (.node 16229 (.node 15083 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9257_ordered : SortedNatTree.ordered row9257 = true := by decide
set_option maxRecDepth 4096 in
theorem row9257_values : SortedNatTree.values row9257 = Data.row9257 := rfl
theorem row9257_checked : SortedNatTree.contains row9257 = Data.row9257.contains := by
  rw [SortedNatTree.contains_eq_list row9257 row9257_ordered, row9257_values]
def row9277 : SortedNatTree.Tree := (.node 14869 (.node 13411 (.node 12301 (.node 9283 .empty .empty) .empty) (.node 13729 .empty .empty)) (.node 16417 (.node 15643 (.node 15013 .empty .empty) .empty) (.node 16729 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row9277_ordered : SortedNatTree.ordered row9277 = true := by decide
set_option maxRecDepth 4096 in
theorem row9277_values : SortedNatTree.values row9277 = Data.row9277 := rfl
theorem row9277_checked : SortedNatTree.contains row9277 = Data.row9277.contains := by
  rw [SortedNatTree.contains_eq_list row9277 row9277_ordered, row9277_values]
def row9281 : SortedNatTree.Tree := (.node 14951 (.node 12821 (.node 10133 (.node 9539 .empty .empty) (.node 12641 .empty .empty)) (.node 14519 (.node 13829 .empty .empty) .empty)) (.node 15581 (.node 15263 (.node 15137 .empty .empty) .empty) (.node 16451 (.node 16073 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9281_ordered : SortedNatTree.ordered row9281 = true := by decide
set_option maxRecDepth 4096 in
theorem row9281_values : SortedNatTree.values row9281 = Data.row9281 := rfl
theorem row9281_checked : SortedNatTree.contains row9281 = Data.row9281.contains := by
  rw [SortedNatTree.contains_eq_list row9281 row9281_ordered, row9281_values]
def row9283 : SortedNatTree.Tree := (.node 12841 (.node 11113 (.node 9631 (.node 9349 .empty .empty) (.node 10399 .empty .empty)) (.node 11497 (.node 11161 .empty .empty) .empty)) (.node 15541 (.node 14449 (.node 14281 .empty .empty) .empty) (.node 16447 (.node 16141 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9283_ordered : SortedNatTree.ordered row9283 = true := by decide
set_option maxRecDepth 4096 in
theorem row9283_values : SortedNatTree.values row9283 = Data.row9283 := rfl
theorem row9283_checked : SortedNatTree.contains row9283 = Data.row9283.contains := by
  rw [SortedNatTree.contains_eq_list row9283 row9283_ordered, row9283_values]
def row9293 : SortedNatTree.Tree := (.node 13241 (.node 12647 (.node 10139 (.node 9311 .empty .empty) (.node 10979 .empty .empty)) (.node 13001 (.node 12899 .empty .empty) .empty)) (.node 14627 (.node 13907 (.node 13337 .empty .empty) .empty) (.node 15671 (.node 15641 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9293_ordered : SortedNatTree.ordered row9293 = true := by decide
set_option maxRecDepth 4096 in
theorem row9293_values : SortedNatTree.values row9293 = Data.row9293 := rfl
theorem row9293_checked : SortedNatTree.contains row9293 = Data.row9293.contains := by
  rw [SortedNatTree.contains_eq_list row9293 row9293_ordered, row9293_values]
def row9311 : SortedNatTree.Tree := (.node 13709 (.node 12641 (.node 11969 (.node 10391 .empty .empty) .empty) (.node 13499 (.node 13331 .empty .empty) .empty)) (.node 15149 (.node 14087 (.node 14081 .empty .empty) .empty) (.node 15401 (.node 15173 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row9311_ordered : SortedNatTree.ordered row9311 = true := by decide
set_option maxRecDepth 4096 in
theorem row9311_values : SortedNatTree.values row9311 = Data.row9311 := rfl
theorem row9311_checked : SortedNatTree.contains row9311 = Data.row9311.contains := by
  rw [SortedNatTree.contains_eq_list row9311 row9311_ordered, row9311_values]
def row9319 : SortedNatTree.Tree := (.node 12829 (.node 11131 (.node 10597 (.node 10513 .empty .empty) (.node 10753 .empty .empty)) (.node 11551 (.node 11173 .empty .empty) (.node 12097 .empty .empty))) (.node 14593 (.node 13399 (.node 13249 .empty .empty) (.node 14407 .empty .empty)) (.node 15661 (.node 15619 .empty .empty) (.node 16057 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9319_ordered : SortedNatTree.ordered row9319 = true := by decide
set_option maxRecDepth 4096 in
theorem row9319_values : SortedNatTree.values row9319 = Data.row9319 := rfl
theorem row9319_checked : SortedNatTree.contains row9319 = Data.row9319.contains := by
  rw [SortedNatTree.contains_eq_list row9319 row9319_ordered, row9319_values]
def row9323 : SortedNatTree.Tree := (.node 13397 (.node 10331 (.node 9551 (.node 9533 (.node 9491 .empty .empty) .empty) (.node 9791 .empty .empty)) (.node 11519 (.node 10781 .empty .empty) (.node 12239 .empty .empty))) (.node 14843 (.node 14081 (.node 13649 .empty .empty) (.node 14549 .empty .empty)) (.node 16421 (.node 15299 .empty .empty) (.node 16547 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row9323_ordered : SortedNatTree.ordered row9323 = true := by decide
set_option maxRecDepth 4096 in
theorem row9323_values : SortedNatTree.values row9323 = Data.row9323 := rfl
theorem row9323_checked : SortedNatTree.contains row9323 = Data.row9323.contains := by
  rw [SortedNatTree.contains_eq_list row9323 row9323_ordered, row9323_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
