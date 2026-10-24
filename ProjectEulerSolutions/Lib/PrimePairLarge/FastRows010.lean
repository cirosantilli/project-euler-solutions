import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row4787 : SortedNatTree.Tree := (.node 10949 (.node 7649 (.node 7013 (.node 5981 (.node 5237 (.node 5099 .empty .empty) .empty) (.node 6047 .empty .empty)) (.node 7229 (.node 7187 .empty .empty) (.node 7433 .empty .empty))) (.node 9929 (.node 9257 (.node 8573 (.node 8009 .empty .empty) .empty) (.node 9371 .empty .empty)) (.node 10781 (.node 10457 .empty .empty) (.node 10889 .empty .empty)))) (.node 17333 (.node 13901 (.node 12893 (.node 12281 (.node 11411 .empty .empty) .empty) (.node 13553 .empty .empty)) (.node 14699 (.node 14087 .empty .empty) (.node 15641 .empty .empty))) (.node 19289 (.node 18593 (.node 17417 (.node 17387 .empty .empty) .empty) (.node 19139 .empty .empty)) (.node 20147 (.node 19751 .empty .empty) (.node 20639 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4787_ordered : SortedNatTree.ordered row4787 = true := by decide
set_option maxRecDepth 4096 in
theorem row4787_values : SortedNatTree.values row4787 = Data.row4787 := rfl
theorem row4787_checked : SortedNatTree.contains row4787 = Data.row4787.contains := by
  rw [SortedNatTree.contains_eq_list row4787 row4787_ordered, row4787_values]
def row4789 : SortedNatTree.Tree := (.node 14431 (.node 10177 (.node 7561 (.node 6547 (.node 5179 (.node 5119 .empty .empty) .empty) (.node 6553 .empty .empty)) (.node 9277 (.node 8269 .empty .empty) (.node 9871 .empty .empty))) (.node 12979 (.node 11113 (.node 11071 (.node 10657 .empty .empty) .empty) (.node 11617 .empty .empty)) (.node 13399 (.node 13147 .empty .empty) (.node 14221 .empty .empty)))) (.node 17599 (.node 17041 (.node 15991 (.node 15667 (.node 14713 .empty .empty) .empty) (.node 16369 .empty .empty)) (.node 17209 (.node 17191 .empty .empty) (.node 17317 .empty .empty))) (.node 18127 (.node 17683 (.node 17623 .empty .empty) (.node 18061 .empty .empty)) (.node 19489 (.node 18301 .empty .empty) (.node 20161 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4789_ordered : SortedNatTree.ordered row4789 = true := by decide
set_option maxRecDepth 4096 in
theorem row4789_values : SortedNatTree.values row4789 = Data.row4789 := rfl
theorem row4789_checked : SortedNatTree.contains row4789 = Data.row4789.contains := by
  rw [SortedNatTree.contains_eq_list row4789 row4789_ordered, row4789_values]
def row4793 : SortedNatTree.Tree := (.node 13007 (.node 7013 (.node 5399 (.node 5003 (.node 4871 .empty .empty) (.node 5099 .empty .empty)) (.node 6389 (.node 5501 .empty .empty) (.node 6869 .empty .empty))) (.node 8783 (.node 8123 (.node 7481 .empty .empty) (.node 8669 .empty .empty)) (.node 11621 (.node 9791 .empty .empty) .empty))) (.node 16889 (.node 16361 (.node 13229 (.node 13151 .empty .empty) (.node 13697 .empty .empty)) (.node 16631 (.node 16427 .empty .empty) .empty)) (.node 19697 (.node 18251 (.node 17393 .empty .empty) (.node 19073 .empty .empty)) (.node 20333 (.node 20327 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4793_ordered : SortedNatTree.ordered row4793 = true := by decide
set_option maxRecDepth 4096 in
theorem row4793_values : SortedNatTree.values row4793 = Data.row4793 := rfl
theorem row4793_checked : SortedNatTree.contains row4793 = Data.row4793.contains := by
  rw [SortedNatTree.contains_eq_list row4793 row4793_ordered, row4793_values]
def row4799 : SortedNatTree.Tree := (.node 11483 (.node 7583 (.node 5639 (.node 5231 (.node 4889 .empty .empty) (.node 5531 .empty .empty)) (.node 6329 (.node 5867 .empty .empty) (.node 7487 .empty .empty))) (.node 10181 (.node 9461 (.node 8597 .empty .empty) (.node 9521 .empty .empty)) (.node 11321 (.node 10253 .empty .empty) (.node 11447 .empty .empty)))) (.node 15377 (.node 13367 (.node 12113 (.node 11489 .empty .empty) (.node 12569 .empty .empty)) (.node 14537 (.node 13613 .empty .empty) (.node 14843 .empty .empty))) (.node 18287 (.node 17189 (.node 16979 .empty .empty) (.node 17489 .empty .empty)) (.node 19421 (.node 18617 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4799_ordered : SortedNatTree.ordered row4799 = true := by decide
set_option maxRecDepth 4096 in
theorem row4799_values : SortedNatTree.values row4799 = Data.row4799 := rfl
theorem row4799_checked : SortedNatTree.contains row4799 = Data.row4799.contains := by
  rw [SortedNatTree.contains_eq_list row4799 row4799_ordered, row4799_values]
def row4801 : SortedNatTree.Tree := (.node 9931 (.node 7159 (.node 6229 (.node 5419 (.node 5197 .empty .empty) (.node 5839 .empty .empty)) (.node 6733 (.node 6361 .empty .empty) (.node 6961 .empty .empty))) (.node 8923 (.node 7993 (.node 7927 .empty .empty) (.node 8317 .empty .empty)) (.node 9397 (.node 9133 .empty .empty) .empty))) (.node 15061 (.node 12211 (.node 12109 (.node 11953 .empty .empty) (.node 12163 .empty .empty)) (.node 12739 (.node 12697 .empty .empty) (.node 14419 .empty .empty))) (.node 18481 (.node 16381 (.node 15919 .empty .empty) (.node 17431 .empty .empty)) (.node 19501 (.node 19249 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4801_ordered : SortedNatTree.ordered row4801 = true := by decide
set_option maxRecDepth 4096 in
theorem row4801_values : SortedNatTree.values row4801 = Data.row4801 := rfl
theorem row4801_checked : SortedNatTree.contains row4801 = Data.row4801.contains := by
  rw [SortedNatTree.contains_eq_list row4801 row4801_ordered, row4801_values]
def row4813 : SortedNatTree.Tree := (.node 12553 (.node 7057 (.node 5869 (.node 5701 (.node 5323 .empty .empty) (.node 5827 .empty .empty)) (.node 6661 (.node 6427 .empty .empty) .empty)) (.node 9643 (.node 8179 (.node 7351 .empty .empty) (.node 8863 .empty .empty)) (.node 12379 (.node 12241 .empty .empty) .empty))) (.node 17047 (.node 15643 (.node 14737 (.node 13687 .empty .empty) (.node 14947 .empty .empty)) (.node 16963 (.node 16573 .empty .empty) .empty)) (.node 19087 (.node 18457 (.node 17257 .empty .empty) (.node 18919 .empty .empty)) (.node 20599 (.node 20107 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4813_ordered : SortedNatTree.ordered row4813 = true := by decide
set_option maxRecDepth 4096 in
theorem row4813_values : SortedNatTree.values row4813 = Data.row4813 := rfl
theorem row4813_checked : SortedNatTree.contains row4813 = Data.row4813.contains := by
  rw [SortedNatTree.contains_eq_list row4813 row4813_ordered, row4813_values]
def row4817 : SortedNatTree.Tree := (.node 14813 (.node 6197 (.node 5783 (.node 5741 (.node 5021 .empty .empty) .empty) (.node 6173 (.node 5903 .empty .empty) .empty)) (.node 13367 (.node 12251 (.node 7127 .empty .empty) .empty) (.node 14633 (.node 13901 .empty .empty) .empty))) (.node 17333 (.node 16619 (.node 15413 (.node 15263 .empty .empty) .empty) (.node 17207 (.node 16691 .empty .empty) .empty)) (.node 18521 (.node 18257 (.node 17417 .empty .empty) .empty) (.node 20639 (.node 18959 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4817_ordered : SortedNatTree.ordered row4817 = true := by decide
set_option maxRecDepth 4096 in
theorem row4817_values : SortedNatTree.values row4817 = Data.row4817 := rfl
theorem row4817_checked : SortedNatTree.contains row4817 = Data.row4817.contains := by
  rw [SortedNatTree.contains_eq_list row4817 row4817_ordered, row4817_values]
def row4831 : SortedNatTree.Tree := (.node 11677 (.node 6967 (.node 6451 (.node 5209 (.node 4951 .empty .empty) (.node 5623 .empty .empty)) (.node 6691 (.node 6553 .empty .empty) .empty)) (.node 8737 (.node 7699 (.node 7069 .empty .empty) (.node 8191 .empty .empty)) (.node 9631 (.node 9181 .empty .empty) .empty))) (.node 17497 (.node 14713 (.node 12841 (.node 12823 .empty .empty) (.node 13567 .empty .empty)) (.node 17359 (.node 16561 .empty .empty) .empty)) (.node 20563 (.node 19993 (.node 17623 .empty .empty) .empty) (.node 21031 (.node 20593 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4831_ordered : SortedNatTree.ordered row4831 = true := by decide
set_option maxRecDepth 4096 in
theorem row4831_values : SortedNatTree.values row4831 = Data.row4831 := rfl
theorem row4831_checked : SortedNatTree.contains row4831 = Data.row4831.contains := by
  rw [SortedNatTree.contains_eq_list row4831 row4831_ordered, row4831_values]
def row4861 : SortedNatTree.Tree := (.node 11779 (.node 6829 (.node 6163 (.node 5689 (.node 5659 .empty .empty) (.node 5953 .empty .empty)) (.node 6361 (.node 6229 .empty .empty) .empty)) (.node 9007 (.node 7573 (.node 7243 .empty .empty) (.node 7879 .empty .empty)) (.node 11317 (.node 10321 .empty .empty) .empty))) (.node 17449 (.node 15439 (.node 12799 (.node 12577 .empty .empty) (.node 14341 .empty .empty)) (.node 16993 (.node 15451 .empty .empty) .empty)) (.node 18541 (.node 18517 (.node 18253 .empty .empty) .empty) (.node 20563 (.node 18637 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4861_ordered : SortedNatTree.ordered row4861 = true := by decide
set_option maxRecDepth 4096 in
theorem row4861_values : SortedNatTree.values row4861 = Data.row4861 := rfl
theorem row4861_checked : SortedNatTree.contains row4861 = Data.row4861.contains := by
  rw [SortedNatTree.contains_eq_list row4861 row4861_ordered, row4861_values]
def row4871 : SortedNatTree.Tree := (.node 10853 (.node 8237 (.node 6959 (.node 6719 (.node 6389 (.node 5237 .empty .empty) .empty) (.node 6947 .empty .empty)) (.node 8069 (.node 7673 (.node 7649 .empty .empty) .empty) (.node 8123 .empty .empty))) (.node 9833 (.node 8861 (.node 8627 (.node 8429 .empty .empty) .empty) (.node 9791 .empty .empty)) (.node 10181 (.node 9887 .empty .empty) (.node 10259 .empty .empty)))) (.node 15287 (.node 13553 (.node 12071 (.node 11423 (.node 10937 .empty .empty) .empty) (.node 12401 .empty .empty)) (.node 13997 (.node 13877 .empty .empty) (.node 14057 .empty .empty))) (.node 19463 (.node 17783 (.node 17333 (.node 16937 .empty .empty) .empty) (.node 19403 .empty .empty)) (.node 20507 (.node 19727 .empty .empty) (.node 21101 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4871_ordered : SortedNatTree.ordered row4871 = true := by decide
set_option maxRecDepth 4096 in
theorem row4871_values : SortedNatTree.values row4871 = Data.row4871 := rfl
theorem row4871_checked : SortedNatTree.contains row4871 = Data.row4871.contains := by
  rw [SortedNatTree.contains_eq_list row4871 row4871_ordered, row4871_values]
def row4877 : SortedNatTree.Tree := (.node 9161 (.node 6947 (.node 5981 (.node 5693 (.node 5501 (.node 5171 .empty .empty) .empty) (.node 5813 (.node 5711 .empty .empty) .empty)) (.node 6551 (.node 6359 (.node 6257 .empty .empty) .empty) (.node 6719 .empty .empty))) (.node 7643 (.node 7349 (.node 7307 (.node 6977 .empty .empty) .empty) (.node 7559 .empty .empty)) (.node 8291 (.node 8231 (.node 8123 .empty .empty) .empty) (.node 8573 .empty .empty)))) (.node 16631 (.node 12809 (.node 11681 (.node 11177 (.node 11087 .empty .empty) .empty) (.node 12143 .empty .empty)) (.node 14891 (.node 14771 (.node 13103 .empty .empty) .empty) (.node 14939 .empty .empty))) (.node 19319 (.node 17729 (.node 17627 (.node 16883 .empty .empty) .empty) (.node 18461 .empty .empty)) (.node 20771 (.node 20627 (.node 20393 .empty .empty) .empty) (.node 20879 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4877_ordered : SortedNatTree.ordered row4877 = true := by decide
set_option maxRecDepth 4096 in
theorem row4877_values : SortedNatTree.values row4877 = Data.row4877 := rfl
theorem row4877_checked : SortedNatTree.contains row4877 = Data.row4877.contains := by
  rw [SortedNatTree.contains_eq_list row4877 row4877_ordered, row4877_values]
def row4889 : SortedNatTree.Tree := (.node 11903 (.node 7643 (.node 6311 (.node 5669 (.node 5333 .empty .empty) (.node 6053 .empty .empty)) (.node 7481 (.node 7187 .empty .empty) .empty)) (.node 9743 (.node 8237 (.node 8123 .empty .empty) .empty) (.node 11597 (.node 9833 .empty .empty) .empty))) (.node 15101 (.node 13781 (.node 12101 (.node 11987 .empty .empty) .empty) (.node 14843 (.node 13883 .empty .empty) .empty)) (.node 17987 (.node 17609 (.node 15137 .empty .empty) .empty) (.node 20249 (.node 18149 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4889_ordered : SortedNatTree.ordered row4889 = true := by decide
set_option maxRecDepth 4096 in
theorem row4889_values : SortedNatTree.values row4889 = Data.row4889 := rfl
theorem row4889_checked : SortedNatTree.contains row4889 = Data.row4889.contains := by
  rw [SortedNatTree.contains_eq_list row4889 row4889_ordered, row4889_values]
def row4903 : SortedNatTree.Tree := (.node 10531 (.node 8839 (.node 6271 (.node 5659 (.node 4933 .empty .empty) (.node 5683 .empty .empty)) (.node 7789 (.node 6991 .empty .empty) (.node 8389 .empty .empty))) (.node 9349 (.node 9007 (.node 8893 .empty .empty) (.node 9337 .empty .empty)) (.node 9811 (.node 9643 .empty .empty) .empty))) (.node 15391 (.node 12379 (.node 12073 (.node 11287 .empty .empty) (.node 12211 .empty .empty)) (.node 13627 (.node 13597 .empty .empty) (.node 14887 .empty .empty))) (.node 17341 (.node 16249 (.node 15493 .empty .empty) (.node 16741 .empty .empty)) (.node 20731 (.node 20359 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4903_ordered : SortedNatTree.ordered row4903 = true := by decide
set_option maxRecDepth 4096 in
theorem row4903_values : SortedNatTree.values row4903 = Data.row4903 := rfl
theorem row4903_checked : SortedNatTree.contains row4903 = Data.row4903.contains := by
  rw [SortedNatTree.contains_eq_list row4903 row4903_ordered, row4903_values]
def row4909 : SortedNatTree.Tree := (.node 10477 (.node 7933 (.node 5647 (.node 5443 (.node 5023 .empty .empty) (.node 5503 .empty .empty)) (.node 7177 (.node 5857 .empty .empty) (.node 7723 .empty .empty))) (.node 9157 (.node 8713 (.node 8293 .empty .empty) (.node 9013 .empty .empty)) (.node 10369 (.node 9907 .empty .empty) .empty))) (.node 14947 (.node 13627 (.node 10957 (.node 10837 .empty .empty) (.node 13441 .empty .empty)) (.node 14821 (.node 14029 .empty .empty) .empty)) (.node 19543 (.node 18859 (.node 17989 .empty .empty) (.node 19237 .empty .empty)) (.node 20719 (.node 20071 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4909_ordered : SortedNatTree.ordered row4909 = true := by decide
set_option maxRecDepth 4096 in
theorem row4909_values : SortedNatTree.values row4909 = Data.row4909 := rfl
theorem row4909_checked : SortedNatTree.contains row4909 = Data.row4909.contains := by
  rw [SortedNatTree.contains_eq_list row4909 row4909_ordered, row4909_values]
def row4919 : SortedNatTree.Tree := (.node 10631 (.node 7607 (.node 6599 (.node 5897 (.node 5189 (.node 5147 .empty .empty) .empty) (.node 6197 .empty .empty)) (.node 7481 (.node 6869 (.node 6803 .empty .empty) .empty) (.node 7559 .empty .empty))) (.node 8783 (.node 8627 (.node 7877 (.node 7829 .empty .empty) .empty) (.node 8741 .empty .empty)) (.node 10061 (.node 9203 .empty .empty) (.node 10163 .empty .empty)))) (.node 13907 (.node 11597 (.node 10781 (.node 10739 (.node 10709 .empty .empty) .empty) (.node 11093 .empty .empty)) (.node 13691 (.node 13337 .empty .empty) (.node 13841 .empty .empty))) (.node 19661 (.node 17351 (.node 17159 (.node 15527 .empty .empty) .empty) (.node 18131 .empty .empty)) (.node 19979 (.node 19889 .empty .empty) (.node 20201 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4919_ordered : SortedNatTree.ordered row4919 = true := by decide
set_option maxRecDepth 4096 in
theorem row4919_values : SortedNatTree.values row4919 = Data.row4919 := rfl
theorem row4919_checked : SortedNatTree.contains row4919 = Data.row4919.contains := by
  rw [SortedNatTree.contains_eq_list row4919 row4919_ordered, row4919_values]
def row4931 : SortedNatTree.Tree := (.node 12953 (.node 8501 (.node 7109 (.node 5843 (.node 5309 (.node 5009 .empty .empty) .empty) (.node 6551 .empty .empty)) (.node 7703 (.node 7349 .empty .empty) (.node 7817 .empty .empty))) (.node 10973 (.node 9833 (.node 9551 .empty .empty) (.node 10253 .empty .empty)) (.node 11681 (.node 11471 .empty .empty) (.node 11933 .empty .empty)))) (.node 15383 (.node 14327 (.node 13457 (.node 13337 .empty .empty) (.node 13487 .empty .empty)) (.node 15017 (.node 14621 .empty .empty) (.node 15137 .empty .empty))) (.node 18701 (.node 17747 (.node 16223 .empty .empty) (.node 17903 .empty .empty)) (.node 19421 (.node 19031 .empty .empty) (.node 20333 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4931_ordered : SortedNatTree.ordered row4931 = true := by decide
set_option maxRecDepth 4096 in
theorem row4931_values : SortedNatTree.values row4931 = Data.row4931 := rfl
theorem row4931_checked : SortedNatTree.contains row4931 = Data.row4931.contains := by
  rw [SortedNatTree.contains_eq_list row4931 row4931_ordered, row4931_values]
def row4933 : SortedNatTree.Tree := (.node 14293 (.node 10009 (.node 8629 (.node 6271 (.node 5701 .empty .empty) (.node 6361 .empty .empty)) (.node 9421 (.node 8893 .empty .empty) .empty)) (.node 13291 (.node 12109 (.node 11131 .empty .empty) (.node 13159 .empty .empty)) (.node 14149 (.node 13669 .empty .empty) .empty))) (.node 17761 (.node 16267 (.node 15643 (.node 15349 .empty .empty) (.node 15727 .empty .empty)) (.node 17257 (.node 17107 .empty .empty) .empty)) (.node 19867 (.node 18517 (.node 17971 .empty .empty) (.node 19441 .empty .empty)) (.node 20071 (.node 19963 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4933_ordered : SortedNatTree.ordered row4933 = true := by decide
set_option maxRecDepth 4096 in
theorem row4933_values : SortedNatTree.values row4933 = Data.row4933 := rfl
theorem row4933_checked : SortedNatTree.contains row4933 = Data.row4933.contains := by
  rw [SortedNatTree.contains_eq_list row4933 row4933_ordered, row4933_values]
def row4937 : SortedNatTree.Tree := (.node 10589 (.node 7649 (.node 7043 (.node 6491 (.node 6053 (.node 5399 .empty .empty) .empty) (.node 7001 .empty .empty)) (.node 7247 (.node 7103 .empty .empty) (.node 7583 .empty .empty))) (.node 9221 (.node 7919 (.node 7841 .empty .empty) (.node 8573 .empty .empty)) (.node 9521 (.node 9419 .empty .empty) (.node 9851 .empty .empty)))) (.node 14831 (.node 13049 (.node 12893 (.node 12161 (.node 11927 .empty .empty) .empty) (.node 12983 .empty .empty)) (.node 13691 (.node 13523 .empty .empty) (.node 14747 .empty .empty))) (.node 17891 (.node 16319 (.node 15569 .empty .empty) (.node 17597 .empty .empty)) (.node 18521 (.node 18257 .empty .empty) (.node 20543 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4937_ordered : SortedNatTree.ordered row4937 = true := by decide
set_option maxRecDepth 4096 in
theorem row4937_values : SortedNatTree.values row4937 = Data.row4937 := rfl
theorem row4937_checked : SortedNatTree.contains row4937 = Data.row4937.contains := by
  rw [SortedNatTree.contains_eq_list row4937 row4937_ordered, row4937_values]
def row4943 : SortedNatTree.Tree := (.node 10709 (.node 6701 (.node 5261 (.node 5039 (.node 5021 .empty .empty) (.node 5189 .empty .empty)) (.node 5741 (.node 5417 .empty .empty) (.node 6449 .empty .empty))) (.node 7919 (.node 7517 (.node 7151 .empty .empty) (.node 7757 .empty .empty)) (.node 10151 (.node 8819 .empty .empty) (.node 10163 .empty .empty)))) (.node 19379 (.node 13877 (.node 11801 (.node 10781 .empty .empty) (.node 13763 .empty .empty)) (.node 17573 (.node 15329 .empty .empty) (.node 18749 .empty .empty))) (.node 20219 (.node 19559 (.node 19391 .empty .empty) (.node 19919 .empty .empty)) (.node 20297 (.node 20261 .empty .empty) (.node 21011 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4943_ordered : SortedNatTree.ordered row4943 = true := by decide
set_option maxRecDepth 4096 in
theorem row4943_values : SortedNatTree.values row4943 = Data.row4943 := rfl
theorem row4943_checked : SortedNatTree.contains row4943 = Data.row4943.contains := by
  rw [SortedNatTree.contains_eq_list row4943 row4943_ordered, row4943_values]
def row4951 : SortedNatTree.Tree := (.node 10453 (.node 8719 (.node 6343 (.node 5563 (.node 5107 .empty .empty) (.node 5821 .empty .empty)) (.node 7537 (.node 6781 .empty .empty) .empty)) (.node 9241 (.node 9049 (.node 8779 .empty .empty) .empty) (.node 9697 (.node 9601 .empty .empty) .empty))) (.node 15307 (.node 13963 (.node 11311 (.node 10723 .empty .empty) (.node 11779 .empty .empty)) (.node 15031 (.node 14029 .empty .empty) .empty)) (.node 16273 (.node 16267 (.node 15349 .empty .empty) .empty) (.node 18451 (.node 18229 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4951_ordered : SortedNatTree.ordered row4951 = true := by decide
set_option maxRecDepth 4096 in
theorem row4951_values : SortedNatTree.values row4951 = Data.row4951 := rfl
theorem row4951_checked : SortedNatTree.contains row4951 = Data.row4951.contains := by
  rw [SortedNatTree.contains_eq_list row4951 row4951_ordered, row4951_values]
def row4957 : SortedNatTree.Tree := (.node 13723 (.node 8623 (.node 6577 (.node 5623 (.node 5569 (.node 5023 .empty .empty) .empty) (.node 5839 .empty .empty)) (.node 8059 (.node 8017 (.node 7603 .empty .empty) .empty) (.node 8233 .empty .empty))) (.node 11941 (.node 9787 (.node 9613 (.node 8929 .empty .empty) .empty) (.node 10723 .empty .empty)) (.node 12697 (.node 12601 .empty .empty) (.node 13441 .empty .empty)))) (.node 17419 (.node 16069 (.node 15607 (.node 15361 (.node 15091 .empty .empty) .empty) (.node 15919 .empty .empty)) (.node 17041 (.node 16267 .empty .empty) (.node 17203 .empty .empty))) (.node 19891 (.node 18523 (.node 18397 (.node 18313 .empty .empty) .empty) (.node 19141 .empty .empty)) (.node 20161 (.node 20047 .empty .empty) (.node 20551 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4957_ordered : SortedNatTree.ordered row4957 = true := by decide
set_option maxRecDepth 4096 in
theorem row4957_values : SortedNatTree.values row4957 = Data.row4957 := rfl
theorem row4957_checked : SortedNatTree.contains row4957 = Data.row4957.contains := by
  rw [SortedNatTree.contains_eq_list row4957 row4957_ordered, row4957_values]
def row4967 : SortedNatTree.Tree := (.node 12539 (.node 7547 (.node 5849 (.node 5531 (.node 5441 (.node 5387 .empty .empty) .empty) (.node 5573 .empty .empty)) (.node 6827 (.node 5939 (.node 5903 .empty .empty) .empty) (.node 7127 .empty .empty))) (.node 8747 (.node 8219 (.node 8117 (.node 7877 .empty .empty) .empty) (.node 8501 .empty .empty)) (.node 11057 (.node 9857 .empty .empty) (.node 11393 .empty .empty)))) (.node 16061 (.node 13577 (.node 13331 (.node 12959 (.node 12911 .empty .empty) .empty) (.node 13499 .empty .empty)) (.node 15467 (.node 15377 .empty .empty) (.node 15473 .empty .empty))) (.node 20051 (.node 19121 (.node 18713 (.node 18149 .empty .empty) .empty) (.node 19463 .empty .empty)) (.node 20771 (.node 20129 .empty .empty) (.node 20897 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4967_ordered : SortedNatTree.ordered row4967 = true := by decide
set_option maxRecDepth 4096 in
theorem row4967_values : SortedNatTree.values row4967 = Data.row4967 := rfl
theorem row4967_checked : SortedNatTree.contains row4967 = Data.row4967.contains := by
  rw [SortedNatTree.contains_eq_list row4967 row4967_ordered, row4967_values]
def row4969 : SortedNatTree.Tree := (.node 11383 (.node 7507 (.node 6133 (.node 5881 (.node 5827 .empty .empty) (.node 6067 .empty .empty)) (.node 6679 (.node 6397 .empty .empty) (.node 7321 .empty .empty))) (.node 9049 (.node 8461 (.node 8329 .empty .empty) (.node 8923 .empty .empty)) (.node 10861 (.node 9817 .empty .empty) (.node 10891 .empty .empty)))) (.node 15817 (.node 13693 (.node 12373 (.node 11497 .empty .empty) (.node 12487 .empty .empty)) (.node 15193 (.node 14293 .empty .empty) (.node 15373 .empty .empty))) (.node 18049 (.node 17401 (.node 16603 .empty .empty) (.node 17791 .empty .empty)) (.node 19471 (.node 19183 .empty .empty) (.node 20959 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4969_ordered : SortedNatTree.ordered row4969 = true := by decide
set_option maxRecDepth 4096 in
theorem row4969_values : SortedNatTree.values row4969 = Data.row4969 := rfl
theorem row4969_checked : SortedNatTree.contains row4969 = Data.row4969.contains := by
  rw [SortedNatTree.contains_eq_list row4969 row4969_ordered, row4969_values]
def row4973 : SortedNatTree.Tree := (.node 13841 (.node 10133 (.node 6569 (.node 6221 (.node 5717 (.node 5303 .empty .empty) .empty) (.node 6389 .empty .empty)) (.node 9413 (.node 7901 (.node 7829 .empty .empty) .empty) (.node 9629 .empty .empty))) (.node 11261 (.node 10601 (.node 10433 (.node 10223 .empty .empty) .empty) (.node 10631 .empty .empty)) (.node 12941 (.node 11699 .empty .empty) (.node 13757 .empty .empty)))) (.node 17783 (.node 16427 (.node 16319 (.node 15629 (.node 14087 .empty .empty) .empty) (.node 16349 .empty .empty)) (.node 17033 (.node 16529 .empty .empty) (.node 17519 .empty .empty))) (.node 19421 (.node 18587 (.node 18329 (.node 17939 .empty .empty) .empty) (.node 19157 .empty .empty)) (.node 19913 (.node 19583 .empty .empty) (.node 20051 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4973_ordered : SortedNatTree.ordered row4973 = true := by decide
set_option maxRecDepth 4096 in
theorem row4973_values : SortedNatTree.values row4973 = Data.row4973 := rfl
theorem row4973_checked : SortedNatTree.contains row4973 = Data.row4973.contains := by
  rw [SortedNatTree.contains_eq_list row4973 row4973_ordered, row4973_values]
def row4987 : SortedNatTree.Tree := (.node 14149 (.node 12583 (.node 8461 (.node 6991 (.node 5197 .empty .empty) .empty) (.node 11497 (.node 8641 .empty .empty) .empty)) (.node 12829 (.node 12763 (.node 12619 .empty .empty) .empty) (.node 13759 .empty .empty))) (.node 18043 (.node 16033 (.node 15991 (.node 14947 .empty .empty) .empty) (.node 16453 .empty .empty)) (.node 18541 (.node 18217 (.node 18097 .empty .empty) .empty) (.node 20899 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row4987_ordered : SortedNatTree.ordered row4987 = true := by decide
set_option maxRecDepth 4096 in
theorem row4987_values : SortedNatTree.values row4987 = Data.row4987 := rfl
theorem row4987_checked : SortedNatTree.contains row4987 = Data.row4987.contains := by
  rw [SortedNatTree.contains_eq_list row4987 row4987_ordered, row4987_values]
def row4993 : SortedNatTree.Tree := (.node 12487 (.node 8563 (.node 6571 (.node 5527 (.node 4999 .empty .empty) (.node 5779 .empty .empty)) (.node 8269 (.node 7561 .empty .empty) (.node 8521 .empty .empty))) (.node 9181 (.node 8971 (.node 8887 .empty .empty) (.node 9133 .empty .empty)) (.node 9661 (.node 9511 .empty .empty) (.node 10369 .empty .empty)))) (.node 16927 (.node 13669 (.node 13183 (.node 12613 .empty .empty) (.node 13567 .empty .empty)) (.node 15199 (.node 14461 .empty .empty) (.node 15373 .empty .empty))) (.node 18223 (.node 17551 (.node 17383 .empty .empty) (.node 18181 .empty .empty)) (.node 19609 (.node 18307 .empty .empty) (.node 20407 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4993_ordered : SortedNatTree.ordered row4993 = true := by decide
set_option maxRecDepth 4096 in
theorem row4993_values : SortedNatTree.values row4993 = Data.row4993 := rfl
theorem row4993_checked : SortedNatTree.contains row4993 = Data.row4993.contains := by
  rw [SortedNatTree.contains_eq_list row4993 row4993_ordered, row4993_values]
def row4999 : SortedNatTree.Tree := (.node 11047 (.node 7069 (.node 6367 (.node 5119 (.node 5107 .empty .empty) (.node 5791 .empty .empty)) (.node 6829 (.node 6763 .empty .empty) .empty)) (.node 10429 (.node 8779 (.node 7309 .empty .empty) (.node 9883 .empty .empty)) (.node 10861 (.node 10837 .empty .empty) .empty))) (.node 15907 (.node 13537 (.node 11701 (.node 11491 .empty .empty) (.node 13159 .empty .empty)) (.node 15817 (.node 14143 .empty .empty) .empty)) (.node 19249 (.node 18427 (.node 16927 .empty .empty) (.node 19219 .empty .empty)) (.node 20143 (.node 19447 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4999_ordered : SortedNatTree.ordered row4999 = true := by decide
set_option maxRecDepth 4096 in
theorem row4999_values : SortedNatTree.values row4999 = Data.row4999 := rfl
theorem row4999_checked : SortedNatTree.contains row4999 = Data.row4999.contains := by
  rw [SortedNatTree.contains_eq_list row4999 row4999_ordered, row4999_values]
def row5003 : SortedNatTree.Tree := (.node 12809 (.node 7541 (.node 6197 (.node 5939 (.node 5087 .empty .empty) (.node 6131 .empty .empty)) (.node 6911 (.node 6551 .empty .empty) (.node 7349 .empty .empty))) (.node 9029 (.node 8147 (.node 7703 .empty .empty) (.node 8423 .empty .empty)) (.node 11423 (.node 10589 .empty .empty) .empty))) (.node 14723 (.node 14051 (.node 13049 (.node 12983 .empty .empty) (.node 13523 .empty .empty)) (.node 14519 (.node 14447 .empty .empty) .empty)) (.node 18047 (.node 17387 (.node 14939 .empty .empty) (.node 17627 .empty .empty)) (.node 18593 (.node 18233 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5003_ordered : SortedNatTree.ordered row5003 = true := by decide
set_option maxRecDepth 4096 in
theorem row5003_values : SortedNatTree.values row5003 = Data.row5003 := rfl
theorem row5003_checked : SortedNatTree.contains row5003 = Data.row5003.contains := by
  rw [SortedNatTree.contains_eq_list row5003 row5003_ordered, row5003_values]
def row5009 : SortedNatTree.Tree := (.node 12959 (.node 7757 (.node 6197 (.node 5861 (.node 5849 (.node 5351 .empty .empty) .empty) (.node 6029 .empty .empty)) (.node 6947 (.node 6353 .empty .empty) (.node 7109 .empty .empty))) (.node 10133 (.node 9059 (.node 8501 (.node 8093 .empty .empty) .empty) (.node 9839 .empty .empty)) (.node 11399 (.node 11057 .empty .empty) (.node 12413 .empty .empty)))) (.node 18233 (.node 15923 (.node 15131 (.node 14669 (.node 13043 .empty .empty) .empty) (.node 15881 .empty .empty)) (.node 16139 (.node 16061 .empty .empty) (.node 16349 .empty .empty))) (.node 19973 (.node 18797 (.node 18461 .empty .empty) (.node 19889 .empty .empty)) (.node 20261 (.node 20183 .empty .empty) (.node 20897 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5009_ordered : SortedNatTree.ordered row5009 = true := by decide
set_option maxRecDepth 4096 in
theorem row5009_values : SortedNatTree.values row5009 = Data.row5009 := rfl
theorem row5009_checked : SortedNatTree.contains row5009 = Data.row5009.contains := by
  rw [SortedNatTree.contains_eq_list row5009 row5009_ordered, row5009_values]
def row5011 : SortedNatTree.Tree := (.node 11383 (.node 7393 (.node 6397 (.node 6163 (.node 5209 .empty .empty) (.node 6271 .empty .empty)) (.node 6961 (.node 6949 .empty .empty) .empty)) (.node 9049 (.node 8191 (.node 8053 .empty .empty) .empty) (.node 10531 (.node 9649 .empty .empty) .empty))) (.node 17107 (.node 14533 (.node 13009 (.node 12823 .empty .empty) (.node 14479 .empty .empty)) (.node 15649 (.node 14827 .empty .empty) .empty)) (.node 19963 (.node 17497 (.node 17377 .empty .empty) .empty) (.node 20947 (.node 20857 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5011_ordered : SortedNatTree.ordered row5011 = true := by decide
set_option maxRecDepth 4096 in
theorem row5011_values : SortedNatTree.values row5011 = Data.row5011 := rfl
theorem row5011_checked : SortedNatTree.contains row5011 = Data.row5011.contains := by
  rw [SortedNatTree.contains_eq_list row5011 row5011_ordered, row5011_values]
def row5021 : SortedNatTree.Tree := (.node 11279 (.node 8219 (.node 5987 (.node 5741 (.node 5507 .empty .empty) .empty) (.node 7127 (.node 6053 .empty .empty) .empty)) (.node 10163 (.node 9497 (.node 8849 .empty .empty) .empty) (.node 10799 (.node 10733 .empty .empty) .empty))) (.node 18311 (.node 16493 (.node 14051 (.node 13109 .empty .empty) .empty) (.node 18059 (.node 17327 .empty .empty) .empty)) (.node 19997 (.node 19697 (.node 18713 .empty .empty) .empty) (.node 20897 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5021_ordered : SortedNatTree.ordered row5021 = true := by decide
set_option maxRecDepth 4096 in
theorem row5021_values : SortedNatTree.values row5021 = Data.row5021 := rfl
theorem row5021_checked : SortedNatTree.contains row5021 = Data.row5021.contains := by
  rw [SortedNatTree.contains_eq_list row5021 row5021_ordered, row5021_values]
def row5023 : SortedNatTree.Tree := (.node 11959 (.node 9769 (.node 7753 (.node 7039 (.node 6841 (.node 5443 .empty .empty) .empty) (.node 7477 .empty .empty)) (.node 8269 (.node 8059 .empty .empty) (.node 8779 .empty .empty))) (.node 10903 (.node 9949 (.node 9907 .empty .empty) (.node 10687 .empty .empty)) (.node 11119 (.node 11071 .empty .empty) (.node 11827 .empty .empty)))) (.node 16651 (.node 14767 (.node 13477 (.node 13177 (.node 12457 .empty .empty) .empty) (.node 13687 .empty .empty)) (.node 16417 (.node 15277 .empty .empty) (.node 16447 .empty .empty))) (.node 18133 (.node 17239 (.node 16993 .empty .empty) (.node 17509 .empty .empty)) (.node 20011 (.node 18211 .empty .empty) (.node 20983 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5023_ordered : SortedNatTree.ordered row5023 = true := by decide
set_option maxRecDepth 4096 in
theorem row5023_values : SortedNatTree.values row5023 = Data.row5023 := rfl
theorem row5023_checked : SortedNatTree.contains row5023 = Data.row5023.contains := by
  rw [SortedNatTree.contains_eq_list row5023 row5023_ordered, row5023_values]
def row5039 : SortedNatTree.Tree := (.node 12107 (.node 7547 (.node 6317 (.node 5669 (.node 5153 .empty .empty) (.node 6299 .empty .empty)) (.node 7001 (.node 6329 .empty .empty) .empty)) (.node 8111 (.node 8093 (.node 7727 .empty .empty) .empty) (.node 9539 (.node 8867 .empty .empty) .empty))) (.node 18341 (.node 16001 (.node 15233 (.node 14897 .empty .empty) (.node 15761 .empty .empty)) (.node 17627 (.node 16661 .empty .empty) .empty)) (.node 19421 (.node 19037 (.node 18521 .empty .empty) .empty) (.node 20201 (.node 20177 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5039_ordered : SortedNatTree.ordered row5039 = true := by decide
set_option maxRecDepth 4096 in
theorem row5039_values : SortedNatTree.values row5039 = Data.row5039 := rfl
theorem row5039_checked : SortedNatTree.contains row5039 = Data.row5039.contains := by
  rw [SortedNatTree.contains_eq_list row5039 row5039_ordered, row5039_values]
def row5051 : SortedNatTree.Tree := (.node 12323 (.node 7949 (.node 6737 (.node 5939 (.node 5903 (.node 5867 .empty .empty) .empty) (.node 6101 .empty .empty)) (.node 7253 (.node 6863 (.node 6833 .empty .empty) .empty) (.node 7283 .empty .empty))) (.node 10391 (.node 9929 (.node 9173 (.node 8969 .empty .empty) .empty) (.node 10271 .empty .empty)) (.node 11681 (.node 11393 .empty .empty) (.node 11699 .empty .empty)))) (.node 17783 (.node 15467 (.node 13997 (.node 12539 (.node 12479 .empty .empty) .empty) (.node 15137 .empty .empty)) (.node 17183 (.node 17117 .empty .empty) (.node 17729 .empty .empty))) (.node 19889 (.node 19469 (.node 18947 (.node 18233 .empty .empty) .empty) (.node 19553 .empty .empty)) (.node 20333 (.node 19973 .empty .empty) (.node 20921 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5051_ordered : SortedNatTree.ordered row5051 = true := by decide
set_option maxRecDepth 4096 in
theorem row5051_values : SortedNatTree.values row5051 = Data.row5051 := rfl
theorem row5051_checked : SortedNatTree.contains row5051 = Data.row5051.contains := by
  rw [SortedNatTree.contains_eq_list row5051 row5051_ordered, row5051_values]
def row5059 : SortedNatTree.Tree := (.node 11299 (.node 9319 (.node 8269 (.node 6691 (.node 5659 (.node 5323 .empty .empty) .empty) (.node 7213 .empty .empty)) (.node 9127 (.node 9007 .empty .empty) (.node 9187 .empty .empty))) (.node 10687 (.node 9649 (.node 9613 .empty .empty) (.node 10039 .empty .empty)) (.node 10729 (.node 10711 .empty .empty) (.node 10789 .empty .empty)))) (.node 16381 (.node 13711 (.node 12907 (.node 12073 (.node 11437 .empty .empty) .empty) (.node 13159 .empty .empty)) (.node 14551 (.node 13921 .empty .empty) (.node 15601 .empty .empty))) (.node 19087 (.node 18223 (.node 16693 .empty .empty) (.node 18313 .empty .empty)) (.node 20047 (.node 19501 .empty .empty) (.node 20707 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5059_ordered : SortedNatTree.ordered row5059 = true := by decide
set_option maxRecDepth 4096 in
theorem row5059_values : SortedNatTree.values row5059 = Data.row5059 := rfl
theorem row5059_checked : SortedNatTree.contains row5059 = Data.row5059.contains := by
  rw [SortedNatTree.contains_eq_list row5059 row5059_ordered, row5059_values]
def row5077 : SortedNatTree.Tree := (.node 11551 (.node 7687 (.node 6367 (.node 6043 (.node 5779 .empty .empty) (.node 6091 .empty .empty)) (.node 7459 (.node 6763 .empty .empty) .empty)) (.node 10141 (.node 9907 (.node 7873 .empty .empty) .empty) (.node 10663 (.node 10459 .empty .empty) .empty))) (.node 18223 (.node 16267 (.node 13267 (.node 13159 .empty .empty) (.node 13879 .empty .empty)) (.node 16843 (.node 16633 .empty .empty) .empty)) (.node 20599 (.node 19507 (.node 18493 .empty .empty) .empty) (.node 20809 (.node 20749 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5077_ordered : SortedNatTree.ordered row5077 = true := by decide
set_option maxRecDepth 4096 in
theorem row5077_values : SortedNatTree.values row5077 = Data.row5077 := rfl
theorem row5077_checked : SortedNatTree.contains row5077 = Data.row5077.contains := by
  rw [SortedNatTree.contains_eq_list row5077 row5077_ordered, row5077_values]
def row5081 : SortedNatTree.Tree := (.node 12809 (.node 8513 (.node 7253 (.node 6971 (.node 5783 .empty .empty) (.node 7211 .empty .empty)) (.node 7817 (.node 7307 .empty .empty) (.node 8291 .empty .empty))) (.node 10193 (.node 9203 (.node 8837 .empty .empty) (.node 10139 .empty .empty)) (.node 12401 (.node 11621 .empty .empty) .empty))) (.node 17657 (.node 13877 (.node 13007 (.node 12821 .empty .empty) (.node 13679 .empty .empty)) (.node 16481 (.node 15809 .empty .empty) (.node 16493 .empty .empty))) (.node 19421 (.node 18329 (.node 17921 .empty .empty) (.node 18749 .empty .empty)) (.node 20903 (.node 20399 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5081_ordered : SortedNatTree.ordered row5081 = true := by decide
set_option maxRecDepth 4096 in
theorem row5081_values : SortedNatTree.values row5081 = Data.row5081 := rfl
theorem row5081_checked : SortedNatTree.contains row5081 = Data.row5081.contains := by
  rw [SortedNatTree.contains_eq_list row5081 row5081_ordered, row5081_values]
def row5087 : SortedNatTree.Tree := (.node 10733 (.node 6779 (.node 5897 (.node 5519 (.node 5273 .empty .empty) (.node 5861 .empty .empty)) (.node 6569 (.node 5927 .empty .empty) (.node 6719 .empty .empty))) (.node 9539 (.node 7247 (.node 6803 .empty .empty) (.node 8147 .empty .empty)) (.node 10463 (.node 10457 .empty .empty) .empty))) (.node 15269 (.node 14411 (.node 12917 (.node 11789 .empty .empty) (.node 13001 .empty .empty)) (.node 15149 (.node 14561 .empty .empty) .empty)) (.node 19391 (.node 17597 (.node 17207 .empty .empty) (.node 19301 .empty .empty)) (.node 20483 (.node 20063 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5087_ordered : SortedNatTree.ordered row5087 = true := by decide
set_option maxRecDepth 4096 in
theorem row5087_values : SortedNatTree.values row5087 = Data.row5087 := rfl
theorem row5087_checked : SortedNatTree.contains row5087 = Data.row5087.contains := by
  rw [SortedNatTree.contains_eq_list row5087 row5087_ordered, row5087_values]
def row5099 : SortedNatTree.Tree := (.node 9467 (.node 7649 (.node 6857 (.node 5501 (.node 5471 .empty .empty) (.node 5507 .empty .empty)) (.node 7523 (.node 7151 .empty .empty) .empty)) (.node 8273 (.node 7949 (.node 7853 .empty .empty) (.node 8231 .empty .empty)) (.node 9323 (.node 8663 .empty .empty) .empty))) (.node 14747 (.node 13619 (.node 11939 (.node 9587 .empty .empty) (.node 13499 .empty .empty)) (.node 14621 (.node 13829 .empty .empty) .empty)) (.node 17477 (.node 17351 (.node 17189 .empty .empty) .empty) (.node 19841 (.node 17909 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5099_ordered : SortedNatTree.ordered row5099 = true := by decide
set_option maxRecDepth 4096 in
theorem row5099_values : SortedNatTree.values row5099 = Data.row5099 := rfl
theorem row5099_checked : SortedNatTree.contains row5099 = Data.row5099.contains := by
  rw [SortedNatTree.contains_eq_list row5099 row5099_ordered, row5099_values]
def row5101 : SortedNatTree.Tree := (.node 14887 (.node 9931 (.node 6883 (.node 6397 (.node 6151 (.node 5953 .empty .empty) .empty) (.node 6571 .empty .empty)) (.node 8263 (.node 7237 (.node 7069 .empty .empty) .empty) (.node 8293 .empty .empty))) (.node 12613 (.node 10831 (.node 10429 (.node 10369 .empty .empty) .empty) (.node 12553 .empty .empty)) (.node 14551 (.node 13789 (.node 12853 .empty .empty) .empty) (.node 14821 .empty .empty)))) (.node 17107 (.node 16141 (.node 15391 (.node 15241 (.node 15031 .empty .empty) .empty) (.node 15913 .empty .empty)) (.node 16843 (.node 16561 (.node 16339 .empty .empty) .empty) (.node 16903 .empty .empty))) (.node 18691 (.node 17737 (.node 17359 (.node 17137 .empty .empty) .empty) (.node 18013 .empty .empty)) (.node 19531 (.node 19501 (.node 19447 .empty .empty) .empty) (.node 20029 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5101_ordered : SortedNatTree.ordered row5101 = true := by decide
set_option maxRecDepth 4096 in
theorem row5101_values : SortedNatTree.values row5101 = Data.row5101 := rfl
theorem row5101_checked : SortedNatTree.contains row5101 = Data.row5101.contains := by
  rw [SortedNatTree.contains_eq_list row5101 row5101_ordered, row5101_values]
def row5107 : SortedNatTree.Tree := (.node 12721 (.node 8311 (.node 6793 (.node 5839 (.node 5779 .empty .empty) (.node 6763 .empty .empty)) (.node 7879 (.node 7717 .empty .empty) (.node 8017 .empty .empty))) (.node 10099 (.node 9403 (.node 8641 .empty .empty) (.node 9901 .empty .empty)) (.node 12373 (.node 10987 .empty .empty) .empty))) (.node 17299 (.node 16033 (.node 14107 (.node 13297 .empty .empty) (.node 14551 .empty .empty)) (.node 16417 (.node 16231 .empty .empty) .empty)) (.node 18379 (.node 17497 (.node 17341 .empty .empty) (.node 17959 .empty .empty)) (.node 20269 (.node 19801 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5107_ordered : SortedNatTree.ordered row5107 = true := by decide
set_option maxRecDepth 4096 in
theorem row5107_values : SortedNatTree.values row5107 = Data.row5107 := rfl
theorem row5107_checked : SortedNatTree.contains row5107 = Data.row5107.contains := by
  rw [SortedNatTree.contains_eq_list row5107 row5107_ordered, row5107_values]
def row5113 : SortedNatTree.Tree := (.node 9967 (.node 7789 (.node 6793 (.node 5449 (.node 5431 .empty .empty) (.node 5659 .empty .empty)) (.node 6991 (.node 6823 .empty .empty) (.node 7069 .empty .empty))) (.node 9001 (.node 8641 (.node 8293 .empty .empty) (.node 8929 .empty .empty)) (.node 9343 (.node 9007 .empty .empty) .empty))) (.node 14767 (.node 13183 (.node 12973 (.node 11113 .empty .empty) (.node 13177 .empty .empty)) (.node 13963 (.node 13513 .empty .empty) (.node 14737 .empty .empty))) (.node 18583 (.node 15601 (.node 15319 .empty .empty) (.node 17707 .empty .empty)) (.node 20899 (.node 18679 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5113_ordered : SortedNatTree.ordered row5113 = true := by decide
set_option maxRecDepth 4096 in
theorem row5113_values : SortedNatTree.values row5113 = Data.row5113 := rfl
theorem row5113_checked : SortedNatTree.contains row5113 = Data.row5113.contains := by
  rw [SortedNatTree.contains_eq_list row5113 row5113_ordered, row5113_values]
def row5119 : SortedNatTree.Tree := (.node 11731 (.node 7459 (.node 6337 (.node 5503 (.node 5233 .empty .empty) (.node 6073 .empty .empty)) (.node 6637 (.node 6529 .empty .empty) (.node 7207 .empty .empty))) (.node 8677 (.node 7621 (.node 7537 .empty .empty) (.node 8647 .empty .empty)) (.node 10177 (.node 9781 .empty .empty) (.node 10957 .empty .empty)))) (.node 18013 (.node 14173 (.node 12601 (.node 12277 .empty .empty) (.node 12853 .empty .empty)) (.node 16963 (.node 14797 .empty .empty) (.node 17137 .empty .empty))) (.node 20089 (.node 18541 (.node 18481 .empty .empty) (.node 19927 .empty .empty)) (.node 20707 (.node 20389 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5119_ordered : SortedNatTree.ordered row5119 = true := by decide
set_option maxRecDepth 4096 in
theorem row5119_values : SortedNatTree.values row5119 = Data.row5119 := rfl
theorem row5119_checked : SortedNatTree.contains row5119 = Data.row5119.contains := by
  rw [SortedNatTree.contains_eq_list row5119 row5119_ordered, row5119_values]
def row5147 : SortedNatTree.Tree := (.node 12227 (.node 7331 (.node 5297 (.node 5273 (.node 5231 .empty .empty) .empty) (.node 6959 (.node 6857 .empty .empty) .empty)) (.node 9203 (.node 7823 (.node 7487 .empty .empty) .empty) (.node 12161 (.node 9851 .empty .empty) .empty))) (.node 15101 (.node 13613 (.node 13259 (.node 13007 .empty .empty) .empty) (.node 14957 (.node 14537 .empty .empty) .empty)) (.node 16067 (.node 15671 (.node 15329 .empty .empty) .empty) (.node 20201 (.node 18047 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5147_ordered : SortedNatTree.ordered row5147 = true := by decide
set_option maxRecDepth 4096 in
theorem row5147_values : SortedNatTree.values row5147 = Data.row5147 := rfl
theorem row5147_checked : SortedNatTree.contains row5147 = Data.row5147.contains := by
  rw [SortedNatTree.contains_eq_list row5147 row5147_ordered, row5147_values]
def row5153 : SortedNatTree.Tree := (.node 10487 (.node 8297 (.node 6101 (.node 5867 (.node 5849 .empty .empty) (.node 6047 .empty .empty)) (.node 7529 (.node 6971 .empty .empty) (.node 7589 .empty .empty))) (.node 9311 (.node 8819 (.node 8693 .empty .empty) (.node 9041 .empty .empty)) (.node 10079 (.node 9491 .empty .empty) .empty))) (.node 14843 (.node 11579 (.node 10949 (.node 10559 .empty .empty) (.node 11087 .empty .empty)) (.node 12251 (.node 11969 .empty .empty) (.node 12923 .empty .empty))) (.node 17657 (.node 15683 (.node 15107 .empty .empty) (.node 16493 .empty .empty)) (.node 19709 (.node 19001 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5153_ordered : SortedNatTree.ordered row5153 = true := by decide
set_option maxRecDepth 4096 in
theorem row5153_values : SortedNatTree.values row5153 = Data.row5153 := rfl
theorem row5153_checked : SortedNatTree.contains row5153 = Data.row5153.contains := by
  rw [SortedNatTree.contains_eq_list row5153 row5153_ordered, row5153_values]
def row5167 : SortedNatTree.Tree := (.node 11887 (.node 9619 (.node 7177 (.node 6661 (.node 5737 .empty .empty) (.node 6967 .empty .empty)) (.node 8761 (.node 7741 .empty .empty) (.node 8941 .empty .empty))) (.node 10837 (.node 10501 (.node 10459 .empty .empty) (.node 10657 .empty .empty)) (.node 11329 (.node 10867 .empty .empty) .empty))) (.node 16561 (.node 15319 (.node 13513 (.node 12799 .empty .empty) (.node 15061 .empty .empty)) (.node 16141 (.node 16087 .empty .empty) .empty)) (.node 19309 (.node 17599 (.node 16633 .empty .empty) (.node 19051 .empty .empty)) (.node 20857 (.node 19447 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5167_ordered : SortedNatTree.ordered row5167 = true := by decide
set_option maxRecDepth 4096 in
theorem row5167_values : SortedNatTree.values row5167 = Data.row5167 := rfl
theorem row5167_checked : SortedNatTree.contains row5167 = Data.row5167.contains := by
  rw [SortedNatTree.contains_eq_list row5167 row5167_ordered, row5167_values]
def row5171 : SortedNatTree.Tree := (.node 11411 (.node 8837 (.node 7187 (.node 5393 (.node 5297 .empty .empty) (.node 5897 .empty .empty)) (.node 8111 (.node 8039 .empty .empty) .empty)) (.node 10211 (.node 10151 (.node 9431 .empty .empty) .empty) (.node 10457 (.node 10247 .empty .empty) .empty))) (.node 15017 (.node 13967 (.node 13679 (.node 13109 .empty .empty) .empty) (.node 14537 (.node 14447 .empty .empty) .empty)) (.node 18713 (.node 18521 (.node 17387 .empty .empty) .empty) (.node 20693 (.node 20231 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5171_ordered : SortedNatTree.ordered row5171 = true := by decide
set_option maxRecDepth 4096 in
theorem row5171_values : SortedNatTree.values row5171 = Data.row5171 := rfl
theorem row5171_checked : SortedNatTree.contains row5171 = Data.row5171.contains := by
  rw [SortedNatTree.contains_eq_list row5171 row5171_ordered, row5171_values]
def row5179 : SortedNatTree.Tree := (.node 11437 (.node 8863 (.node 6763 (.node 5851 (.node 5683 .empty .empty) (.node 6301 .empty .empty)) (.node 8209 (.node 7873 .empty .empty) .empty)) (.node 9619 (.node 9337 (.node 9043 .empty .empty) (.node 9439 .empty .empty)) (.node 11287 (.node 9661 .empty .empty) .empty))) (.node 17419 (.node 14083 (.node 12841 (.node 11491 .empty .empty) (.node 14029 .empty .empty)) (.node 17317 (.node 15859 .empty .empty) .empty)) (.node 20269 (.node 19069 (.node 19009 .empty .empty) (.node 20023 .empty .empty)) (.node 20707 (.node 20593 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5179_ordered : SortedNatTree.ordered row5179 = true := by decide
set_option maxRecDepth 4096 in
theorem row5179_values : SortedNatTree.values row5179 = Data.row5179 := rfl
theorem row5179_checked : SortedNatTree.contains row5179 = Data.row5179.contains := by
  rw [SortedNatTree.contains_eq_list row5179 row5179_ordered, row5179_values]
def row5189 : SortedNatTree.Tree := (.node 14813 (.node 10313 (.node 9209 (.node 6491 (.node 6329 .empty .empty) (.node 8573 .empty .empty)) (.node 9539 (.node 9377 .empty .empty) .empty)) (.node 12671 (.node 11909 (.node 10331 .empty .empty) (.node 12473 .empty .empty)) (.node 14741 (.node 14723 .empty .empty) .empty))) (.node 17291 (.node 16529 (.node 15461 (.node 15017 .empty .empty) (.node 15671 .empty .empty)) (.node 16889 (.node 16553 .empty .empty) .empty)) (.node 19553 (.node 18089 (.node 17327 .empty .empty) (.node 18731 .empty .empty)) (.node 20063 (.node 19727 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5189_ordered : SortedNatTree.ordered row5189 = true := by decide
set_option maxRecDepth 4096 in
theorem row5189_values : SortedNatTree.values row5189 = Data.row5189 := rfl
theorem row5189_checked : SortedNatTree.contains row5189 = Data.row5189.contains := by
  rw [SortedNatTree.contains_eq_list row5189 row5189_ordered, row5189_values]
def row5197 : SortedNatTree.Tree := (.node 12289 (.node 8389 (.node 7591 (.node 6733 (.node 5701 .empty .empty) (.node 6997 .empty .empty)) (.node 8209 (.node 7963 .empty .empty) (.node 8221 .empty .empty))) (.node 10657 (.node 8929 (.node 8629 .empty .empty) (.node 9679 .empty .empty)) (.node 10831 (.node 10729 .empty .empty) (.node 11443 .empty .empty)))) (.node 15451 (.node 14341 (.node 12421 (.node 12409 .empty .empty) (.node 13933 .empty .empty)) (.node 15013 (.node 14653 .empty .empty) (.node 15277 .empty .empty))) (.node 19441 (.node 18553 (.node 17791 .empty .empty) (.node 19381 .empty .empty)) (.node 20347 (.node 20101 .empty .empty) (.node 20599 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5197_ordered : SortedNatTree.ordered row5197 = true := by decide
set_option maxRecDepth 4096 in
theorem row5197_values : SortedNatTree.values row5197 = Data.row5197 := rfl
theorem row5197_checked : SortedNatTree.contains row5197 = Data.row5197.contains := by
  rw [SortedNatTree.contains_eq_list row5197 row5197_ordered, row5197_values]
def row5209 : SortedNatTree.Tree := (.node 11959 (.node 7963 (.node 7207 (.node 6397 (.node 5419 .empty .empty) (.node 7057 .empty .empty)) (.node 7879 (.node 7417 .empty .empty) .empty)) (.node 10069 (.node 8677 (.node 8191 .empty .empty) (.node 9241 .empty .empty)) (.node 11329 (.node 10303 .empty .empty) .empty))) (.node 15973 (.node 14533 (.node 12823 (.node 11971 .empty .empty) (.node 13873 .empty .empty)) (.node 15331 (.node 15091 .empty .empty) .empty)) (.node 18757 (.node 18253 (.node 18127 .empty .empty) (.node 18427 .empty .empty)) (.node 20161 (.node 19429 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5209_ordered : SortedNatTree.ordered row5209 = true := by decide
set_option maxRecDepth 4096 in
theorem row5209_values : SortedNatTree.values row5209 = Data.row5209 := rfl
theorem row5209_checked : SortedNatTree.contains row5209 = Data.row5209.contains := by
  rw [SortedNatTree.contains_eq_list row5209 row5209_ordered, row5209_values]
def row5227 : SortedNatTree.Tree := (.node 12517 (.node 8641 (.node 6481 (.node 6229 (.node 5683 .empty .empty) (.node 6379 .empty .empty)) (.node 7591 (.node 7243 .empty .empty) (.node 8353 .empty .empty))) (.node 9973 (.node 9613 (.node 9067 .empty .empty) (.node 9817 .empty .empty)) (.node 10093 (.node 10039 .empty .empty) (.node 11941 .empty .empty)))) (.node 16921 (.node 13669 (.node 13183 (.node 13159 .empty .empty) (.node 13441 .empty .empty)) (.node 14251 (.node 14029 .empty .empty) (.node 15889 .empty .empty))) (.node 18013 (.node 17509 (.node 17137 .empty .empty) (.node 17551 .empty .empty)) (.node 19333 (.node 18541 .empty .empty) (.node 20749 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5227_ordered : SortedNatTree.ordered row5227 = true := by decide
set_option maxRecDepth 4096 in
theorem row5227_values : SortedNatTree.values row5227 = Data.row5227 := rfl
theorem row5227_checked : SortedNatTree.contains row5227 = Data.row5227.contains := by
  rw [SortedNatTree.contains_eq_list row5227 row5227_ordered, row5227_values]
def row5231 : SortedNatTree.Tree := (.node 12983 (.node 9833 (.node 7949 (.node 6581 (.node 5843 (.node 5387 .empty .empty) .empty) (.node 7589 .empty .empty)) (.node 9437 (.node 9431 .empty .empty) (.node 9539 .empty .empty))) (.node 10949 (.node 10193 (.node 9941 .empty .empty) (.node 10691 .empty .empty)) (.node 12041 (.node 11549 .empty .empty) (.node 12377 .empty .empty)))) (.node 17117 (.node 15377 (.node 15017 (.node 13967 (.node 13229 .empty .empty) .empty) (.node 15101 .empty .empty)) (.node 16193 (.node 15401 .empty .empty) (.node 17093 .empty .empty))) (.node 19301 (.node 17477 (.node 17351 .empty .empty) (.node 18143 .empty .empty)) (.node 19991 (.node 19433 .empty .empty) (.node 20249 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5231_ordered : SortedNatTree.ordered row5231 = true := by decide
set_option maxRecDepth 4096 in
theorem row5231_values : SortedNatTree.values row5231 = Data.row5231 := rfl
theorem row5231_checked : SortedNatTree.contains row5231 = Data.row5231.contains := by
  rw [SortedNatTree.contains_eq_list row5231 row5231_ordered, row5231_values]
def row5233 : SortedNatTree.Tree := (.node 9829 (.node 7309 (.node 6367 (.node 5779 (.node 5323 .empty .empty) (.node 6343 .empty .empty)) (.node 6451 (.node 6373 .empty .empty) .empty)) (.node 8677 (.node 8089 (.node 7639 .empty .empty) .empty) (.node 9649 (.node 9601 .empty .empty) .empty))) (.node 13597 (.node 13477 (.node 11317 (.node 10729 .empty .empty) (.node 12373 .empty .empty)) (.node 13591 (.node 13513 .empty .empty) .empty)) (.node 19441 (.node 16111 (.node 14767 .empty .empty) .empty) (.node 20359 (.node 19681 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5233_ordered : SortedNatTree.ordered row5233 = true := by decide
set_option maxRecDepth 4096 in
theorem row5233_values : SortedNatTree.values row5233 = Data.row5233 := rfl
theorem row5233_checked : SortedNatTree.contains row5233 = Data.row5233.contains := by
  rw [SortedNatTree.contains_eq_list row5233 row5233_ordered, row5233_values]
def row5237 : SortedNatTree.Tree := (.node 12197 (.node 7487 (.node 6761 (.node 5783 (.node 5693 (.node 5471 .empty .empty) .empty) (.node 6563 .empty .empty)) (.node 7151 (.node 7043 .empty .empty) (.node 7349 .empty .empty))) (.node 10079 (.node 8537 (.node 8243 (.node 8009 .empty .empty) .empty) (.node 9221 .empty .empty)) (.node 10937 (.node 10133 .empty .empty) (.node 12071 .empty .empty)))) (.node 15161 (.node 14447 (.node 13103 (.node 12611 (.node 12281 .empty .empty) .empty) (.node 13217 .empty .empty)) (.node 15053 (.node 14657 .empty .empty) (.node 15077 .empty .empty))) (.node 18287 (.node 17609 (.node 16979 (.node 16451 .empty .empty) .empty) (.node 17921 .empty .empty)) (.node 19457 (.node 18719 .empty .empty) (.node 20117 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5237_ordered : SortedNatTree.ordered row5237 = true := by decide
set_option maxRecDepth 4096 in
theorem row5237_values : SortedNatTree.values row5237 = Data.row5237 := rfl
theorem row5237_checked : SortedNatTree.contains row5237 = Data.row5237.contains := by
  rw [SortedNatTree.contains_eq_list row5237 row5237_ordered, row5237_values]
def row5261 : SortedNatTree.Tree := (.node 10253 (.node 8387 (.node 6779 (.node 5987 (.node 5573 .empty .empty) (.node 6329 .empty .empty)) (.node 7247 (.node 6983 .empty .empty) .empty)) (.node 9689 (.node 9473 (.node 8831 .empty .empty) (.node 9629 .empty .empty)) (.node 10223 (.node 9887 .empty .empty) .empty))) (.node 15797 (.node 12239 (.node 11783 (.node 10973 .empty .empty) (.node 12149 .empty .empty)) (.node 14879 (.node 14207 .empty .empty) .empty)) (.node 19037 (.node 16433 (.node 16427 .empty .empty) (.node 18869 .empty .empty)) (.node 20129 (.node 19793 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5261_ordered : SortedNatTree.ordered row5261 = true := by decide
set_option maxRecDepth 4096 in
theorem row5261_values : SortedNatTree.values row5261 = Data.row5261 := rfl
theorem row5261_checked : SortedNatTree.contains row5261 = Data.row5261.contains := by
  rw [SortedNatTree.contains_eq_list row5261 row5261_ordered, row5261_values]
def row5273 : SortedNatTree.Tree := (.node 14057 (.node 8429 (.node 7127 (.node 5651 (.node 5381 .empty .empty) (.node 6833 .empty .empty)) (.node 8093 (.node 7211 .empty .empty) .empty)) (.node 12809 (.node 10631 (.node 8999 .empty .empty) .empty) (.node 13679 (.node 13043 .empty .empty) .empty))) (.node 17477 (.node 15461 (.node 14369 (.node 14159 .empty .empty) (.node 14897 .empty .empty)) (.node 17183 (.node 16619 .empty .empty) .empty)) (.node 19469 (.node 17789 (.node 17597 .empty .empty) .empty) (.node 20477 (.node 20441 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5273_ordered : SortedNatTree.ordered row5273 = true := by decide
set_option maxRecDepth 4096 in
theorem row5273_values : SortedNatTree.values row5273 = Data.row5273 := rfl
theorem row5273_checked : SortedNatTree.contains row5273 = Data.row5273.contains := by
  rw [SortedNatTree.contains_eq_list row5273 row5273_ordered, row5273_values]
def row5279 : SortedNatTree.Tree := (.node 12503 (.node 7793 (.node 6719 (.node 6113 (.node 5783 (.node 5651 .empty .empty) .empty) (.node 6449 .empty .empty)) (.node 7109 (.node 6827 .empty .empty) (.node 7253 .empty .empty))) (.node 9209 (.node 8429 (.node 8231 (.node 7883 .empty .empty) .empty) (.node 8849 .empty .empty)) (.node 12323 (.node 10433 .empty .empty) (.node 12377 .empty .empty)))) (.node 17483 (.node 15299 (.node 15107 (.node 14249 (.node 13829 .empty .empty) .empty) (.node 15287 .empty .empty)) (.node 15749 (.node 15473 .empty .empty) (.node 16979 .empty .empty))) (.node 19583 (.node 17987 (.node 17669 .empty .empty) (.node 18143 .empty .empty)) (.node 19919 (.node 19709 .empty .empty) (.node 20693 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5279_ordered : SortedNatTree.ordered row5279 = true := by decide
set_option maxRecDepth 4096 in
theorem row5279_values : SortedNatTree.values row5279 = Data.row5279 := rfl
theorem row5279_checked : SortedNatTree.contains row5279 = Data.row5279.contains := by
  rw [SortedNatTree.contains_eq_list row5279 row5279_ordered, row5279_values]
def row5281 : SortedNatTree.Tree := (.node 11863 (.node 7951 (.node 6619 (.node 5869 (.node 5437 .empty .empty) (.node 6121 .empty .empty)) (.node 7489 (.node 7159 .empty .empty) (.node 7873 .empty .empty))) (.node 10369 (.node 9001 (.node 8287 .empty .empty) (.node 9433 .empty .empty)) (.node 10939 (.node 10861 .empty .empty) .empty))) (.node 16453 (.node 14107 (.node 12721 (.node 12703 .empty .empty) (.node 13249 .empty .empty)) (.node 15493 (.node 15199 .empty .empty) (.node 16063 .empty .empty))) (.node 19219 (.node 16987 (.node 16573 .empty .empty) (.node 18169 .empty .empty)) (.node 20443 (.node 19597 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5281_ordered : SortedNatTree.ordered row5281 = true := by decide
set_option maxRecDepth 4096 in
theorem row5281_values : SortedNatTree.values row5281 = Data.row5281 := rfl
theorem row5281_checked : SortedNatTree.contains row5281 = Data.row5281.contains := by
  rw [SortedNatTree.contains_eq_list row5281 row5281_ordered, row5281_values]
def row5297 : SortedNatTree.Tree := (.node 14621 (.node 8297 (.node 6263 (.node 6257 (.node 5843 .empty .empty) .empty) (.node 7541 (.node 6701 .empty .empty) .empty)) (.node 11411 (.node 10271 (.node 9791 .empty .empty) .empty) (.node 14369 (.node 11783 .empty .empty) .empty))) (.node 17729 (.node 16763 (.node 16007 (.node 15749 .empty .empty) .empty) (.node 17579 (.node 17333 .empty .empty) .empty)) (.node 18713 (.node 18443 (.node 18311 .empty .empty) .empty) (.node 20441 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5297_ordered : SortedNatTree.ordered row5297 = true := by decide
set_option maxRecDepth 4096 in
theorem row5297_values : SortedNatTree.values row5297 = Data.row5297 := rfl
theorem row5297_checked : SortedNatTree.contains row5297 = Data.row5297.contains := by
  rw [SortedNatTree.contains_eq_list row5297 row5297_ordered, row5297_values]
def row5303 : SortedNatTree.Tree := (.node 12197 (.node 8969 (.node 8243 (.node 7823 (.node 6101 (.node 5849 .empty .empty) .empty) (.node 8123 .empty .empty)) (.node 8597 (.node 8387 .empty .empty) (.node 8681 .empty .empty))) (.node 10181 (.node 9719 (.node 9341 .empty .empty) (.node 10061 .empty .empty)) (.node 10607 (.node 10301 .empty .empty) (.node 11027 .empty .empty)))) (.node 14723 (.node 13457 (.node 12917 (.node 12479 .empty .empty) (.node 13337 .empty .empty)) (.node 14249 (.node 13877 .empty .empty) (.node 14591 .empty .empty))) (.node 15767 (.node 14843 (.node 14831 .empty .empty) (.node 14891 .empty .empty)) (.node 16487 (.node 16067 .empty .empty) (.node 18617 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5303_ordered : SortedNatTree.ordered row5303 = true := by decide
set_option maxRecDepth 4096 in
theorem row5303_values : SortedNatTree.values row5303 = Data.row5303 := rfl
theorem row5303_checked : SortedNatTree.contains row5303 = Data.row5303.contains := by
  rw [SortedNatTree.contains_eq_list row5303 row5303_ordered, row5303_values]
def row5309 : SortedNatTree.Tree := (.node 10427 (.node 7481 (.node 6359 (.node 6011 (.node 5501 .empty .empty) (.node 6257 .empty .empty)) (.node 7151 (.node 6977 .empty .empty) (.node 7283 .empty .empty))) (.node 9413 (.node 8147 (.node 8117 .empty .empty) (.node 9173 .empty .empty)) (.node 10181 (.node 9743 .empty .empty) .empty))) (.node 17657 (.node 14717 (.node 10631 (.node 10463 .empty .empty) (.node 13151 .empty .empty)) (.node 15761 (.node 15551 .empty .empty) (.node 16091 .empty .empty))) (.node 19889 (.node 18371 (.node 17987 .empty .empty) (.node 18917 .empty .empty)) (.node 20549 (.node 20543 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5309_ordered : SortedNatTree.ordered row5309 = true := by decide
set_option maxRecDepth 4096 in
theorem row5309_values : SortedNatTree.values row5309 = Data.row5309 := rfl
theorem row5309_checked : SortedNatTree.contains row5309 = Data.row5309.contains := by
  rw [SortedNatTree.contains_eq_list row5309 row5309_ordered, row5309_values]
def row5323 : SortedNatTree.Tree := (.node 11251 (.node 6397 (.node 5779 (.node 5527 (.node 5521 .empty .empty) (.node 5659 .empty .empty)) (.node 6067 (.node 6037 .empty .empty) (.node 6361 .empty .empty))) (.node 7537 (.node 7237 (.node 6481 .empty .empty) (.node 7333 .empty .empty)) (.node 10357 (.node 8089 .empty .empty) (.node 10657 .empty .empty)))) (.node 16747 (.node 14593 (.node 12889 (.node 11953 .empty .empty) (.node 13759 .empty .empty)) (.node 16111 (.node 15439 .empty .empty) (.node 16453 .empty .empty))) (.node 19387 (.node 18427 (.node 17257 .empty .empty) (.node 19213 .empty .empty)) (.node 20071 (.node 20023 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5323_ordered : SortedNatTree.ordered row5323 = true := by decide
set_option maxRecDepth 4096 in
theorem row5323_values : SortedNatTree.values row5323 = Data.row5323 := rfl
theorem row5323_checked : SortedNatTree.contains row5323 = Data.row5323.contains := by
  rw [SortedNatTree.contains_eq_list row5323 row5323_ordered, row5323_values]
def row5333 : SortedNatTree.Tree := (.node 10781 (.node 8009 (.node 7649 (.node 6719 (.node 5657 .empty .empty) (.node 7283 .empty .empty)) (.node 7901 (.node 7703 .empty .empty) .empty)) (.node 9491 (.node 8543 (.node 8093 .empty .empty) (.node 9029 .empty .empty)) (.node 9887 (.node 9521 .empty .empty) .empty))) (.node 16103 (.node 15383 (.node 12893 (.node 11483 .empty .empty) (.node 13901 .empty .empty)) (.node 15737 (.node 15683 .empty .empty) .empty)) (.node 16703 (.node 16649 (.node 16481 .empty .empty) .empty) (.node 20639 (.node 16829 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5333_ordered : SortedNatTree.ordered row5333 = true := by decide
set_option maxRecDepth 4096 in
theorem row5333_values : SortedNatTree.values row5333 = Data.row5333 := rfl
theorem row5333_checked : SortedNatTree.contains row5333 = Data.row5333.contains := by
  rw [SortedNatTree.contains_eq_list row5333 row5333_ordered, row5333_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
