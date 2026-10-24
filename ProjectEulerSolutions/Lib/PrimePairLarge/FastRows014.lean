import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row6991 : SortedNatTree.Tree := (.node 12823 (.node 9511 (.node 9277 (.node 8923 (.node 8011 .empty .empty) .empty) (.node 9403 .empty .empty)) (.node 12343 (.node 10513 (.node 9697 .empty .empty) .empty) (.node 12703 .empty .empty))) (.node 15331 (.node 14929 (.node 14887 (.node 13291 .empty .empty) .empty) (.node 15217 .empty .empty)) (.node 18439 (.node 17851 .empty .empty) (.node 18493 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6991_ordered : SortedNatTree.ordered row6991 = true := by decide
set_option maxRecDepth 4096 in
theorem row6991_values : SortedNatTree.values row6991 = Data.row6991 := rfl
theorem row6991_checked : SortedNatTree.contains row6991 = Data.row6991.contains := by
  rw [SortedNatTree.contains_eq_list row6991 row6991_ordered, row6991_values]
def row6997 : SortedNatTree.Tree := (.node 13669 (.node 8599 (.node 7507 (.node 7351 (.node 7213 .empty .empty) .empty) (.node 8011 (.node 7879 .empty .empty) .empty)) (.node 12619 (.node 10831 (.node 9901 .empty .empty) .empty) (.node 12967 .empty .empty))) (.node 15031 (.node 14737 (.node 14731 (.node 13879 .empty .empty) .empty) (.node 14887 .empty .empty)) (.node 17449 (.node 16747 (.node 15193 .empty .empty) .empty) (.node 17827 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6997_ordered : SortedNatTree.ordered row6997 = true := by decide
set_option maxRecDepth 4096 in
theorem row6997_values : SortedNatTree.values row6997 = Data.row6997 := rfl
theorem row6997_checked : SortedNatTree.contains row6997 = Data.row6997.contains := by
  rw [SortedNatTree.contains_eq_list row6997 row6997_ordered, row6997_values]
def row7001 : SortedNatTree.Tree := (.node 13451 (.node 11717 (.node 9239 (.node 7853 (.node 7559 .empty .empty) .empty) (.node 9377 .empty .empty)) (.node 12251 (.node 12239 .empty .empty) (.node 13049 .empty .empty))) (.node 14843 (.node 13901 (.node 13829 .empty .empty) (.node 14249 .empty .empty)) (.node 16607 (.node 16349 .empty .empty) (.node 16829 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7001_ordered : SortedNatTree.ordered row7001 = true := by decide
set_option maxRecDepth 4096 in
theorem row7001_values : SortedNatTree.values row7001 = Data.row7001 := rfl
theorem row7001_checked : SortedNatTree.contains row7001 = Data.row7001.contains := by
  rw [SortedNatTree.contains_eq_list row7001 row7001_ordered, row7001_values]
def row7013 : SortedNatTree.Tree := (.node 13103 (.node 8867 (.node 8093 (.node 7517 (.node 7211 .empty .empty) .empty) (.node 8543 (.node 8363 .empty .empty) .empty)) (.node 9497 (.node 9413 (.node 9227 .empty .empty) .empty) (.node 10937 (.node 10067 .empty .empty) .empty))) (.node 16349 (.node 13763 (.node 13751 (.node 13163 .empty .empty) .empty) (.node 14813 (.node 14543 .empty .empty) .empty)) (.node 17489 (.node 17387 (.node 16481 .empty .empty) .empty) (.node 18593 (.node 18077 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7013_ordered : SortedNatTree.ordered row7013 = true := by decide
set_option maxRecDepth 4096 in
theorem row7013_values : SortedNatTree.values row7013 = Data.row7013 := rfl
theorem row7013_checked : SortedNatTree.contains row7013 = Data.row7013.contains := by
  rw [SortedNatTree.contains_eq_list row7013 row7013_ordered, row7013_values]
def row7019 : SortedNatTree.Tree := (.node 14159 (.node 10247 (.node 7877 (.node 7703 (.node 7451 .empty .empty) .empty) (.node 10193 .empty .empty)) (.node 11261 (.node 11117 (.node 10883 .empty .empty) .empty) (.node 11369 .empty .empty))) (.node 16889 (.node 15413 (.node 14657 (.node 14489 .empty .empty) .empty) (.node 15881 .empty .empty)) (.node 18773 (.node 16931 .empty .empty) (.node 18839 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7019_ordered : SortedNatTree.ordered row7019 = true := by decide
set_option maxRecDepth 4096 in
theorem row7019_values : SortedNatTree.values row7019 = Data.row7019 := rfl
theorem row7019_checked : SortedNatTree.contains row7019 = Data.row7019.contains := by
  rw [SortedNatTree.contains_eq_list row7019 row7019_ordered, row7019_values]
def row7027 : SortedNatTree.Tree := (.node 12589 (.node 8821 (.node 8233 (.node 7687 (.node 7297 .empty .empty) .empty) (.node 8623 (.node 8431 .empty .empty) .empty)) (.node 9733 (.node 9127 (.node 8863 .empty .empty) .empty) (.node 12457 (.node 10009 .empty .empty) .empty))) (.node 16657 (.node 14983 (.node 14923 (.node 13567 .empty .empty) .empty) (.node 15427 (.node 15217 .empty .empty) .empty)) (.node 17497 (.node 17203 (.node 16699 .empty .empty) .empty) (.node 18661 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7027_ordered : SortedNatTree.ordered row7027 = true := by decide
set_option maxRecDepth 4096 in
theorem row7027_values : SortedNatTree.values row7027 = Data.row7027 := rfl
theorem row7027_checked : SortedNatTree.contains row7027 = Data.row7027.contains := by
  rw [SortedNatTree.contains_eq_list row7027 row7027_ordered, row7027_values]
def row7039 : SortedNatTree.Tree := (.node 12919 (.node 12007 (.node 9187 (.node 7669 (.node 7177 .empty .empty) .empty) (.node 9391 .empty .empty)) (.node 12577 (.node 12289 .empty .empty) (.node 12637 .empty .empty))) (.node 17257 (.node 15271 (.node 14821 (.node 13933 .empty .empty) .empty) (.node 15649 .empty .empty)) (.node 17707 (.node 17299 .empty .empty) (.node 17791 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7039_ordered : SortedNatTree.ordered row7039 = true := by decide
set_option maxRecDepth 4096 in
theorem row7039_values : SortedNatTree.values row7039 = Data.row7039 := rfl
theorem row7039_checked : SortedNatTree.contains row7039 = Data.row7039.contains := by
  rw [SortedNatTree.contains_eq_list row7039 row7039_ordered, row7039_values]
def row7043 : SortedNatTree.Tree := (.node 13877 (.node 9221 (.node 7589 (.node 7559 (.node 7253 .empty .empty) .empty) (.node 8609 (.node 7949 .empty .empty) .empty)) (.node 11159 (.node 9587 (.node 9437 .empty .empty) .empty) (.node 12263 (.node 12149 .empty .empty) .empty))) (.node 17483 (.node 16703 (.node 15761 (.node 14867 .empty .empty) .empty) (.node 17093 (.node 17027 .empty .empty) .empty)) (.node 18413 (.node 18311 (.node 17807 .empty .empty) .empty) (.node 18701 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7043_ordered : SortedNatTree.ordered row7043 = true := by decide
set_option maxRecDepth 4096 in
theorem row7043_values : SortedNatTree.values row7043 = Data.row7043 := rfl
theorem row7043_checked : SortedNatTree.contains row7043 = Data.row7043.contains := by
  rw [SortedNatTree.contains_eq_list row7043 row7043_ordered, row7043_values]
def row7057 : SortedNatTree.Tree := (.node 11593 (.node 9829 (.node 8887 (.node 8737 (.node 8353 .empty .empty) .empty) (.node 9151 (.node 9109 .empty .empty) .empty)) (.node 10627 (.node 10513 (.node 10039 .empty .empty) .empty) (.node 11119 (.node 10861 .empty .empty) .empty))) (.node 16741 (.node 13093 (.node 11833 (.node 11827 .empty .empty) .empty) (.node 15919 (.node 14029 .empty .empty) .empty)) (.node 17203 (.node 17137 (.node 16993 .empty .empty) .empty) (.node 18253 (.node 17839 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7057_ordered : SortedNatTree.ordered row7057 = true := by decide
set_option maxRecDepth 4096 in
theorem row7057_values : SortedNatTree.values row7057 = Data.row7057 := rfl
theorem row7057_checked : SortedNatTree.contains row7057 = Data.row7057.contains := by
  rw [SortedNatTree.contains_eq_list row7057 row7057_ordered, row7057_values]
def row7069 : SortedNatTree.Tree := (.node 12589 (.node 8707 (.node 7741 (.node 7369 .empty .empty) (.node 8161 .empty .empty)) (.node 9973 (.node 9007 .empty .empty) (.node 10837 .empty .empty))) (.node 15733 (.node 14347 (.node 13147 .empty .empty) (.node 15061 .empty .empty)) (.node 17389 (.node 16057 .empty .empty) (.node 18481 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7069_ordered : SortedNatTree.ordered row7069 = true := by decide
set_option maxRecDepth 4096 in
theorem row7069_values : SortedNatTree.values row7069 = Data.row7069 := rfl
theorem row7069_checked : SortedNatTree.contains row7069 = Data.row7069.contains := by
  rw [SortedNatTree.contains_eq_list row7069 row7069_ordered, row7069_values]
def row7079 : SortedNatTree.Tree := (.node 13487 (.node 10589 (.node 8363 (.node 8111 (.node 7229 .empty .empty) .empty) (.node 9923 (.node 9413 .empty .empty) .empty)) (.node 12119 (.node 10739 (.node 10613 .empty .empty) .empty) (.node 12899 (.node 12263 .empty .empty) .empty))) (.node 17393 (.node 14957 (.node 14177 (.node 13901 .empty .empty) .empty) (.node 16553 (.node 15077 .empty .empty) .empty)) (.node 18719 (.node 18041 (.node 17681 .empty .empty) .empty) (.node 18773 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7079_ordered : SortedNatTree.ordered row7079 = true := by decide
set_option maxRecDepth 4096 in
theorem row7079_values : SortedNatTree.values row7079 = Data.row7079 := rfl
theorem row7079_checked : SortedNatTree.contains row7079 = Data.row7079.contains := by
  rw [SortedNatTree.contains_eq_list row7079 row7079_ordered, row7079_values]
def row7103 : SortedNatTree.Tree := (.node 11933 (.node 9851 (.node 8963 (.node 8273 (.node 7283 .empty .empty) (.node 8447 .empty .empty)) (.node 9539 (.node 9257 .empty .empty) .empty)) (.node 11321 (.node 11159 (.node 10529 .empty .empty) .empty) (.node 11807 (.node 11699 .empty .empty) .empty))) (.node 15731 (.node 13883 (.node 13421 (.node 12809 .empty .empty) .empty) (.node 15443 (.node 14447 .empty .empty) .empty)) (.node 17351 (.node 17327 (.node 17117 .empty .empty) .empty) (.node 18143 (.node 17891 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7103_ordered : SortedNatTree.ordered row7103 = true := by decide
set_option maxRecDepth 4096 in
theorem row7103_values : SortedNatTree.values row7103 = Data.row7103 := rfl
theorem row7103_checked : SortedNatTree.contains row7103 = Data.row7103.contains := by
  rw [SortedNatTree.contains_eq_list row7103 row7103_ordered, row7103_values]
def row7109 : SortedNatTree.Tree := (.node 11057 (.node 9227 (.node 8669 (.node 7841 (.node 7127 .empty .empty) (.node 7907 .empty .empty)) (.node 9011 (.node 8999 .empty .empty) .empty)) (.node 10091 (.node 9839 (.node 9629 .empty .empty) .empty) (.node 10181 (.node 10103 .empty .empty) .empty))) (.node 15641 (.node 14543 (.node 12281 (.node 11813 .empty .empty) (.node 13913 .empty .empty)) (.node 15629 (.node 15377 .empty .empty) .empty)) (.node 17957 (.node 17099 (.node 16889 .empty .empty) .empty) (.node 18701 (.node 18539 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7109_ordered : SortedNatTree.ordered row7109 = true := by decide
set_option maxRecDepth 4096 in
theorem row7109_values : SortedNatTree.values row7109 = Data.row7109 := rfl
theorem row7109_checked : SortedNatTree.contains row7109 = Data.row7109.contains := by
  rw [SortedNatTree.contains_eq_list row7109 row7109_ordered, row7109_values]
def row7121 : SortedNatTree.Tree := (.node 12161 (.node 10691 (.node 8447 (.node 8429 (.node 8297 .empty .empty) .empty) (.node 9689 (.node 9209 .empty .empty) .empty)) (.node 11927 (.node 11177 (.node 11171 .empty .empty) .empty) (.node 12119 .empty .empty))) (.node 16073 (.node 15749 (.node 14159 (.node 13151 .empty .empty) .empty) (.node 15773 .empty .empty)) (.node 18059 (.node 17579 (.node 17327 .empty .empty) .empty) (.node 18521 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7121_ordered : SortedNatTree.ordered row7121 = true := by decide
set_option maxRecDepth 4096 in
theorem row7121_values : SortedNatTree.values row7121 = Data.row7121 := rfl
theorem row7121_checked : SortedNatTree.contains row7121 = Data.row7121.contains := by
  rw [SortedNatTree.contains_eq_list row7121 row7121_ordered, row7121_values]
def row7127 : SortedNatTree.Tree := (.node 13877 (.node 11579 (.node 8849 (.node 7793 (.node 7757 .empty .empty) (.node 8429 .empty .empty)) (.node 11003 (.node 9311 .empty .empty) .empty)) (.node 12251 (.node 12197 (.node 11789 .empty .empty) .empty) (.node 13163 (.node 12281 .empty .empty) .empty))) (.node 15581 (.node 14753 (.node 14153 (.node 13913 .empty .empty) (.node 14447 .empty .empty)) (.node 15383 (.node 15077 .empty .empty) .empty)) (.node 16481 (.node 16349 (.node 16001 .empty .empty) .empty) (.node 18287 (.node 17483 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7127_ordered : SortedNatTree.ordered row7127 = true := by decide
set_option maxRecDepth 4096 in
theorem row7127_values : SortedNatTree.values row7127 = Data.row7127 := rfl
theorem row7127_checked : SortedNatTree.contains row7127 = Data.row7127.contains := by
  rw [SortedNatTree.contains_eq_list row7127 row7127_ordered, row7127_values]
def row7129 : SortedNatTree.Tree := (.node 13099 (.node 10651 (.node 9733 (.node 8893 (.node 7411 .empty .empty) .empty) (.node 10399 (.node 9883 .empty .empty) .empty)) (.node 11827 (.node 11701 (.node 11587 .empty .empty) .empty) (.node 12841 .empty .empty))) (.node 16189 (.node 15241 (.node 14887 (.node 14107 .empty .empty) .empty) (.node 15937 (.node 15661 .empty .empty) .empty)) (.node 17707 (.node 17299 (.node 16693 .empty .empty) .empty) (.node 17923 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7129_ordered : SortedNatTree.ordered row7129 = true := by decide
set_option maxRecDepth 4096 in
theorem row7129_values : SortedNatTree.values row7129 = Data.row7129 := rfl
theorem row7129_checked : SortedNatTree.contains row7129 = Data.row7129.contains := by
  rw [SortedNatTree.contains_eq_list row7129 row7129_ordered, row7129_values]
def row7151 : SortedNatTree.Tree := (.node 11897 (.node 9209 (.node 7529 (.node 7211 .empty .empty) (.node 8081 .empty .empty)) (.node 11777 (.node 11471 .empty .empty) .empty)) (.node 14831 (.node 12911 (.node 12569 .empty .empty) .empty) (.node 15881 (.node 15299 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7151_ordered : SortedNatTree.ordered row7151 = true := by decide
set_option maxRecDepth 4096 in
theorem row7151_values : SortedNatTree.values row7151 = Data.row7151 := rfl
theorem row7151_checked : SortedNatTree.contains row7151 = Data.row7151.contains := by
  rw [SortedNatTree.contains_eq_list row7151 row7151_ordered, row7151_values]
def row7159 : SortedNatTree.Tree := (.node 10141 (.node 8353 (.node 7411 (.node 7219 (.node 7213 .empty .empty) .empty) (.node 7717 (.node 7639 .empty .empty) .empty)) (.node 9133 (.node 8599 (.node 8521 .empty .empty) .empty) (.node 10099 (.node 9883 .empty .empty) .empty))) (.node 16339 (.node 10453 (.node 10333 (.node 10159 .empty .empty) .empty) (.node 15583 (.node 14197 .empty .empty) .empty)) (.node 17047 (.node 16963 (.node 16693 .empty .empty) .empty) (.node 18787 (.node 17737 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7159_ordered : SortedNatTree.ordered row7159 = true := by decide
set_option maxRecDepth 4096 in
theorem row7159_values : SortedNatTree.values row7159 = Data.row7159 := rfl
theorem row7159_checked : SortedNatTree.contains row7159 = Data.row7159.contains := by
  rw [SortedNatTree.contains_eq_list row7159 row7159_ordered, row7159_values]
def row7177 : SortedNatTree.Tree := (.node 13099 (.node 8923 (.node 7933 (.node 7309 (.node 7207 .empty .empty) (.node 7393 .empty .empty)) (.node 8887 (.node 8737 .empty .empty) .empty)) (.node 11971 (.node 9349 (.node 9319 .empty .empty) (.node 11257 .empty .empty)) (.node 12301 (.node 12277 .empty .empty) .empty))) (.node 16981 (.node 14869 (.node 13441 (.node 13291 .empty .empty) (.node 13669 .empty .empty)) (.node 16921 (.node 16567 .empty .empty) .empty)) (.node 17443 (.node 17431 (.node 17137 .empty .empty) .empty) (.node 18439 (.node 18097 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7177_ordered : SortedNatTree.ordered row7177 = true := by decide
set_option maxRecDepth 4096 in
theorem row7177_values : SortedNatTree.values row7177 = Data.row7177 := rfl
theorem row7177_checked : SortedNatTree.contains row7177 = Data.row7177.contains := by
  rw [SortedNatTree.contains_eq_list row7177 row7177_ordered, row7177_values]
def row7187 : SortedNatTree.Tree := (.node 13799 (.node 9491 (.node 8111 (.node 7331 (.node 7229 .empty .empty) .empty) (.node 8171 .empty .empty)) (.node 10781 (.node 10247 .empty .empty) (.node 12143 .empty .empty))) (.node 16787 (.node 14891 (.node 14033 (.node 13841 .empty .empty) .empty) (.node 15581 .empty .empty)) (.node 17939 (.node 17033 .empty .empty) (.node 18059 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7187_ordered : SortedNatTree.ordered row7187 = true := by decide
set_option maxRecDepth 4096 in
theorem row7187_values : SortedNatTree.values row7187 = Data.row7187 := rfl
theorem row7187_checked : SortedNatTree.contains row7187 = Data.row7187.contains := by
  rw [SortedNatTree.contains_eq_list row7187 row7187_ordered, row7187_values]
def row7193 : SortedNatTree.Tree := (.node 13499 (.node 9221 (.node 8429 (.node 7757 (.node 7451 .empty .empty) (.node 7841 .empty .empty)) (.node 8753 (.node 8597 .empty .empty) .empty)) (.node 10589 (.node 9941 (.node 9677 .empty .empty) (.node 10337 .empty .empty)) (.node 12107 (.node 11399 .empty .empty) .empty))) (.node 15761 (.node 14159 (.node 13967 (.node 13697 .empty .empty) (.node 14087 .empty .empty)) (.node 14831 (.node 14633 .empty .empty) .empty)) (.node 17519 (.node 17477 (.node 16139 .empty .empty) .empty) (.node 18401 (.node 18287 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7193_ordered : SortedNatTree.ordered row7193 = true := by decide
set_option maxRecDepth 4096 in
theorem row7193_values : SortedNatTree.values row7193 = Data.row7193 := rfl
theorem row7193_checked : SortedNatTree.contains row7193 = Data.row7193.contains := by
  rw [SortedNatTree.contains_eq_list row7193 row7193_ordered, row7193_values]
def row7207 : SortedNatTree.Tree := (.node 13309 (.node 9391 (.node 8443 (.node 8059 (.node 7927 .empty .empty) (.node 8179 .empty .empty)) (.node 9277 (.node 8863 .empty .empty) .empty)) (.node 11503 (.node 11443 (.node 10651 .empty .empty) .empty) (.node 12163 (.node 11833 .empty .empty) .empty))) (.node 16273 (.node 14887 (.node 14557 (.node 13711 .empty .empty) (.node 14767 .empty .empty)) (.node 15439 (.node 15319 .empty .empty) .empty)) (.node 17209 (.node 16651 (.node 16363 .empty .empty) .empty) (.node 17467 (.node 17341 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7207_ordered : SortedNatTree.ordered row7207 = true := by decide
set_option maxRecDepth 4096 in
theorem row7207_values : SortedNatTree.values row7207 = Data.row7207 := rfl
theorem row7207_checked : SortedNatTree.contains row7207 = Data.row7207.contains := by
  rw [SortedNatTree.contains_eq_list row7207 row7207_ordered, row7207_values]
def row7211 : SortedNatTree.Tree := (.node 13241 (.node 9941 (.node 9323 (.node 8999 (.node 7853 .empty .empty) .empty) (.node 9689 .empty .empty)) (.node 10289 (.node 10223 .empty .empty) (.node 12569 .empty .empty))) (.node 16943 (.node 14813 (.node 14249 .empty .empty) (.node 15959 .empty .empty)) (.node 17669 (.node 17207 .empty .empty) (.node 18143 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7211_ordered : SortedNatTree.ordered row7211 = true := by decide
set_option maxRecDepth 4096 in
theorem row7211_values : SortedNatTree.values row7211 = Data.row7211 := rfl
theorem row7211_checked : SortedNatTree.contains row7211 = Data.row7211.contains := by
  rw [SortedNatTree.contains_eq_list row7211 row7211_ordered, row7211_values]
def row7213 : SortedNatTree.Tree := (.node 10789 (.node 9439 (.node 8713 (.node 7669 (.node 7411 .empty .empty) .empty) (.node 9349 (.node 9343 .empty .empty) .empty)) (.node 10399 (.node 10099 (.node 9601 .empty .empty) .empty) (.node 10687 .empty .empty))) (.node 14173 (.node 11923 (.node 11527 (.node 10903 .empty .empty) .empty) (.node 12721 (.node 11959 .empty .empty) .empty)) (.node 16747 (.node 16417 (.node 15649 .empty .empty) .empty) (.node 18457 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7213_ordered : SortedNatTree.ordered row7213 = true := by decide
set_option maxRecDepth 4096 in
theorem row7213_values : SortedNatTree.values row7213 = Data.row7213 := rfl
theorem row7213_checked : SortedNatTree.contains row7213 = Data.row7213.contains := by
  rw [SortedNatTree.contains_eq_list row7213 row7213_ordered, row7213_values]
def row7219 : SortedNatTree.Tree := (.node 9739 (.node 8377 (.node 8209 (.node 7669 (.node 7537 .empty .empty) .empty) (.node 8317 .empty .empty)) (.node 8923 (.node 8521 .empty .empty) (.node 9619 .empty .empty))) (.node 13063 (.node 10303 (.node 10141 .empty .empty) (.node 11317 .empty .empty)) (.node 15787 (.node 13099 .empty .empty) (.node 17827 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7219_ordered : SortedNatTree.ordered row7219 = true := by decide
set_option maxRecDepth 4096 in
theorem row7219_values : SortedNatTree.values row7219 = Data.row7219 := rfl
theorem row7219_checked : SortedNatTree.contains row7219 = Data.row7219.contains := by
  rw [SortedNatTree.contains_eq_list row7219 row7219_ordered, row7219_values]
def row7229 : SortedNatTree.Tree := (.node 9467 (.node 7937 (.node 7691 (.node 7541 (.node 7307 .empty .empty) .empty) (.node 7829 .empty .empty)) (.node 8693 (.node 8147 .empty .empty) (.node 9257 .empty .empty))) (.node 14537 (.node 11087 (.node 10247 .empty .empty) (.node 13229 .empty .empty)) (.node 16889 (.node 16547 .empty .empty) (.node 18803 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7229_ordered : SortedNatTree.ordered row7229 = true := by decide
set_option maxRecDepth 4096 in
theorem row7229_values : SortedNatTree.values row7229 = Data.row7229 := rfl
theorem row7229_checked : SortedNatTree.contains row7229 = Data.row7229.contains := by
  rw [SortedNatTree.contains_eq_list row7229 row7229_ordered, row7229_values]
def row7237 : SortedNatTree.Tree := (.node 12037 (.node 10093 (.node 7699 (.node 7561 (.node 7507 .empty .empty) .empty) (.node 10069 (.node 7759 .empty .empty) .empty)) (.node 10711 (.node 10567 (.node 10321 .empty .empty) .empty) (.node 11497 (.node 11383 .empty .empty) .empty))) (.node 15427 (.node 13063 (.node 12799 (.node 12589 .empty .empty) .empty) (.node 15361 (.node 13381 .empty .empty) .empty)) (.node 17107 (.node 16267 (.node 16033 .empty .empty) .empty) (.node 18397 (.node 17737 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7237_ordered : SortedNatTree.ordered row7237 = true := by decide
set_option maxRecDepth 4096 in
theorem row7237_values : SortedNatTree.values row7237 = Data.row7237 := rfl
theorem row7237_checked : SortedNatTree.contains row7237 = Data.row7237.contains := by
  rw [SortedNatTree.contains_eq_list row7237 row7237_ordered, row7237_values]
def row7243 : SortedNatTree.Tree := (.node 12517 (.node 10429 (.node 8941 (.node 8713 (.node 8209 .empty .empty) .empty) (.node 9421 .empty .empty)) (.node 11839 (.node 11779 (.node 11131 .empty .empty) .empty) (.node 12451 .empty .empty))) (.node 15973 (.node 14797 (.node 14083 (.node 13669 .empty .empty) .empty) (.node 15643 .empty .empty)) (.node 18583 (.node 18523 (.node 17299 .empty .empty) .empty) (.node 18787 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7243_ordered : SortedNatTree.ordered row7243 = true := by decide
set_option maxRecDepth 4096 in
theorem row7243_values : SortedNatTree.values row7243 = Data.row7243 := rfl
theorem row7243_checked : SortedNatTree.contains row7243 = Data.row7243.contains := by
  rw [SortedNatTree.contains_eq_list row7243 row7243_ordered, row7243_values]
def row7247 : SortedNatTree.Tree := (.node 12479 (.node 9749 (.node 8849 (.node 7541 (.node 7331 .empty .empty) (.node 8111 .empty .empty)) (.node 9377 (.node 9203 .empty .empty) (.node 9689 .empty .empty))) (.node 10709 (.node 10211 (.node 9791 .empty .empty) (.node 10247 .empty .empty)) (.node 11909 (.node 11159 .empty .empty) (.node 12119 .empty .empty)))) (.node 14867 (.node 13397 (.node 13151 (.node 12689 .empty .empty) (.node 13217 .empty .empty)) (.node 13901 (.node 13721 .empty .empty) (.node 14747 .empty .empty))) (.node 16127 (.node 15161 (.node 14957 .empty .empty) (.node 15791 .empty .empty)) (.node 17159 (.node 16631 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7247_ordered : SortedNatTree.ordered row7247 = true := by decide
set_option maxRecDepth 4096 in
theorem row7247_values : SortedNatTree.values row7247 = Data.row7247 := rfl
theorem row7247_checked : SortedNatTree.contains row7247 = Data.row7247.contains := by
  rw [SortedNatTree.contains_eq_list row7247 row7247_ordered, row7247_values]
def row7253 : SortedNatTree.Tree := (.node 13859 (.node 11801 (.node 8243 (.node 8009 (.node 7499 .empty .empty) .empty) (.node 9227 .empty .empty)) (.node 12911 (.node 12197 (.node 11981 .empty .empty) .empty) (.node 13709 .empty .empty))) (.node 16871 (.node 15467 (.node 15263 (.node 14519 .empty .empty) .empty) (.node 16673 .empty .empty)) (.node 17669 (.node 17093 (.node 16979 .empty .empty) .empty) (.node 17909 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7253_ordered : SortedNatTree.ordered row7253 = true := by decide
set_option maxRecDepth 4096 in
theorem row7253_values : SortedNatTree.values row7253 = Data.row7253 := rfl
theorem row7253_checked : SortedNatTree.contains row7253 = Data.row7253.contains := by
  rw [SortedNatTree.contains_eq_list row7253 row7253_ordered, row7253_values]
def row7283 : SortedNatTree.Tree := (.node 9419 (.node 8147 (.node 7901 (.node 7649 (.node 7577 .empty .empty) .empty) (.node 7949 .empty .empty)) (.node 8573 (.node 8537 (.node 8243 .empty .empty) .empty) (.node 9257 .empty .empty))) (.node 14243 (.node 12437 (.node 11783 (.node 9743 .empty .empty) .empty) (.node 13841 .empty .empty)) (.node 17957 (.node 17021 (.node 14537 .empty .empty) .empty) (.node 18287 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7283_ordered : SortedNatTree.ordered row7283 = true := by decide
set_option maxRecDepth 4096 in
theorem row7283_values : SortedNatTree.values row7283 = Data.row7283 := rfl
theorem row7283_checked : SortedNatTree.contains row7283 = Data.row7283.contains := by
  rw [SortedNatTree.contains_eq_list row7283 row7283_ordered, row7283_values]
def row7297 : SortedNatTree.Tree := (.node 10321 (.node 8887 (.node 8179 (.node 7951 (.node 7507 .empty .empty) (.node 8053 .empty .empty)) (.node 8539 (.node 8377 .empty .empty) (.node 8707 .empty .empty))) (.node 9187 (.node 9001 (.node 8923 .empty .empty) (.node 9091 .empty .empty)) (.node 9547 (.node 9421 .empty .empty) .empty))) (.node 14731 (.node 13513 (.node 11839 (.node 10357 .empty .empty) (.node 12829 .empty .empty)) (.node 14149 (.node 13687 .empty .empty) (.node 14431 .empty .empty))) (.node 15643 (.node 15307 (.node 14929 .empty .empty) (.node 15601 .empty .empty)) (.node 17551 (.node 15901 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7297_ordered : SortedNatTree.ordered row7297 = true := by decide
set_option maxRecDepth 4096 in
theorem row7297_values : SortedNatTree.values row7297 = Data.row7297 := rfl
theorem row7297_checked : SortedNatTree.contains row7297 = Data.row7297.contains := by
  rw [SortedNatTree.contains_eq_list row7297 row7297_ordered, row7297_values]
def row7307 : SortedNatTree.Tree := (.node 10163 (.node 9239 (.node 8783 (.node 8387 (.node 7547 .empty .empty) .empty) (.node 8867 .empty .empty)) (.node 9887 (.node 9371 (.node 9323 .empty .empty) .empty) (.node 10067 .empty .empty))) (.node 15443 (.node 12527 (.node 11903 (.node 11351 .empty .empty) .empty) (.node 13553 .empty .empty)) (.node 15761 (.node 15569 .empty .empty) (.node 16061 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7307_ordered : SortedNatTree.ordered row7307 = true := by decide
set_option maxRecDepth 4096 in
theorem row7307_values : SortedNatTree.values row7307 = Data.row7307 := rfl
theorem row7307_checked : SortedNatTree.contains row7307 = Data.row7307.contains := by
  rw [SortedNatTree.contains_eq_list row7307 row7307_ordered, row7307_values]
def row7309 : SortedNatTree.Tree := (.node 11593 (.node 9181 (.node 8563 (.node 7993 (.node 7963 .empty .empty) (.node 8269 .empty .empty)) (.node 9001 (.node 8863 .empty .empty) .empty)) (.node 10627 (.node 9661 (.node 9283 .empty .empty) (.node 10459 .empty .empty)) (.node 11383 (.node 11353 .empty .empty) .empty))) (.node 15121 (.node 13411 (.node 11887 (.node 11827 .empty .empty) (.node 12211 .empty .empty)) (.node 14851 (.node 13693 .empty .empty) .empty)) (.node 16993 (.node 16519 (.node 15907 .empty .empty) .empty) (.node 18229 (.node 17581 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7309_ordered : SortedNatTree.ordered row7309 = true := by decide
set_option maxRecDepth 4096 in
theorem row7309_values : SortedNatTree.values row7309 = Data.row7309 := rfl
theorem row7309_checked : SortedNatTree.contains row7309 = Data.row7309.contains := by
  rw [SortedNatTree.contains_eq_list row7309 row7309_ordered, row7309_values]
def row7321 : SortedNatTree.Tree := (.node 11329 (.node 9439 (.node 8287 (.node 8179 (.node 7369 .empty .empty) .empty) (.node 8737 .empty .empty)) (.node 10513 (.node 9829 .empty .empty) (.node 10531 .empty .empty))) (.node 17377 (.node 14797 (.node 14389 (.node 12421 .empty .empty) .empty) (.node 16729 .empty .empty)) (.node 17977 (.node 17497 .empty .empty) (.node 17989 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7321_ordered : SortedNatTree.ordered row7321 = true := by decide
set_option maxRecDepth 4096 in
theorem row7321_values : SortedNatTree.values row7321 = Data.row7321 := rfl
theorem row7321_checked : SortedNatTree.contains row7321 = Data.row7321.contains := by
  rw [SortedNatTree.contains_eq_list row7321 row7321_ordered, row7321_values]
def row7331 : SortedNatTree.Tree := (.node 12101 (.node 8951 (.node 8513 (.node 8369 (.node 7487 .empty .empty) .empty) (.node 8831 .empty .empty)) (.node 10739 (.node 9227 (.node 9137 .empty .empty) .empty) (.node 11069 .empty .empty))) (.node 15083 (.node 14741 (.node 13691 (.node 12713 .empty .empty) .empty) (.node 14747 .empty .empty)) (.node 17123 (.node 15107 .empty .empty) (.node 18593 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7331_ordered : SortedNatTree.ordered row7331 = true := by decide
set_option maxRecDepth 4096 in
theorem row7331_values : SortedNatTree.values row7331 = Data.row7331 := rfl
theorem row7331_checked : SortedNatTree.contains row7331 = Data.row7331.contains := by
  rw [SortedNatTree.contains_eq_list row7331 row7331_ordered, row7331_values]
def row7333 : SortedNatTree.Tree := (.node 13723 (.node 9151 (.node 8059 (.node 7753 (.node 7393 .empty .empty) .empty) (.node 9103 .empty .empty)) (.node 12373 (.node 11959 (.node 11071 .empty .empty) .empty) (.node 13093 .empty .empty))) (.node 15823 (.node 15073 (.node 15031 (.node 14947 .empty .empty) .empty) (.node 15727 .empty .empty)) (.node 18127 (.node 18013 (.node 17839 .empty .empty) .empty) (.node 18181 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7333_ordered : SortedNatTree.ordered row7333 = true := by decide
set_option maxRecDepth 4096 in
theorem row7333_values : SortedNatTree.values row7333 = Data.row7333 := rfl
theorem row7333_checked : SortedNatTree.contains row7333 = Data.row7333.contains := by
  rw [SortedNatTree.contains_eq_list row7333 row7333_ordered, row7333_values]
def row7349 : SortedNatTree.Tree := (.node 11981 (.node 9521 (.node 8699 (.node 8681 (.node 8147 .empty .empty) .empty) (.node 8753 .empty .empty)) (.node 10151 (.node 9689 .empty .empty) (.node 11117 .empty .empty))) (.node 15737 (.node 13037 (.node 12653 (.node 12149 .empty .empty) .empty) (.node 14969 .empty .empty)) (.node 17627 (.node 16649 .empty .empty) (.node 18077 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7349_ordered : SortedNatTree.ordered row7349 = true := by decide
set_option maxRecDepth 4096 in
theorem row7349_values : SortedNatTree.values row7349 = Data.row7349 := rfl
theorem row7349_checked : SortedNatTree.contains row7349 = Data.row7349.contains := by
  rw [SortedNatTree.contains_eq_list row7349 row7349_ordered, row7349_values]
def row7351 : SortedNatTree.Tree := (.node 12253 (.node 9613 (.node 8287 (.node 7723 (.node 7699 .empty .empty) .empty) (.node 8521 .empty .empty)) (.node 10093 (.node 9739 (.node 9619 .empty .empty) .empty) (.node 11677 .empty .empty))) (.node 14683 (.node 13147 (.node 12979 (.node 12613 .empty .empty) .empty) (.node 13411 .empty .empty)) (.node 17497 (.node 15217 (.node 14851 .empty .empty) .empty) (.node 18061 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7351_ordered : SortedNatTree.ordered row7351 = true := by decide
set_option maxRecDepth 4096 in
theorem row7351_values : SortedNatTree.values row7351 = Data.row7351 := rfl
theorem row7351_checked : SortedNatTree.contains row7351 = Data.row7351.contains := by
  rw [SortedNatTree.contains_eq_list row7351 row7351_ordered, row7351_values]
def row7369 : SortedNatTree.Tree := (.node 11839 (.node 9931 (.node 8629 (.node 8353 (.node 7489 .empty .empty) .empty) (.node 8839 .empty .empty)) (.node 11173 (.node 10267 (.node 9949 .empty .empty) .empty) (.node 11443 .empty .empty))) (.node 15073 (.node 13999 (.node 13963 (.node 13327 .empty .empty) .empty) (.node 14407 .empty .empty)) (.node 17341 (.node 16603 .empty .empty) (.node 17551 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7369_ordered : SortedNatTree.ordered row7369 = true := by decide
set_option maxRecDepth 4096 in
theorem row7369_values : SortedNatTree.values row7369 = Data.row7369 := rfl
theorem row7369_checked : SortedNatTree.contains row7369 = Data.row7369.contains := by
  rw [SortedNatTree.contains_eq_list row7369 row7369_ordered, row7369_values]
def row7393 : SortedNatTree.Tree := (.node 14281 (.node 11173 (.node 9109 (.node 7867 (.node 7687 .empty .empty) .empty) (.node 10177 .empty .empty)) (.node 12043 (.node 11383 .empty .empty) (.node 13033 .empty .empty))) (.node 16843 (.node 15919 (.node 15877 (.node 15583 .empty .empty) .empty) (.node 16381 .empty .empty)) (.node 18043 (.node 17431 .empty .empty) (.node 18637 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7393_ordered : SortedNatTree.ordered row7393 = true := by decide
set_option maxRecDepth 4096 in
theorem row7393_values : SortedNatTree.values row7393 = Data.row7393 := rfl
theorem row7393_checked : SortedNatTree.contains row7393 = Data.row7393.contains := by
  rw [SortedNatTree.contains_eq_list row7393 row7393_ordered, row7393_values]
def row7411 : SortedNatTree.Tree := (.node 13669 (.node 10093 (.node 9043 (.node 8329 .empty .empty) (.node 9973 .empty .empty)) (.node 10939 (.node 10303 .empty .empty) (.node 13309 .empty .empty))) (.node 17029 (.node 14797 (.node 13831 .empty .empty) (.node 15373 .empty .empty)) (.node 18013 (.node 17851 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7411_ordered : SortedNatTree.ordered row7411 = true := by decide
set_option maxRecDepth 4096 in
theorem row7411_values : SortedNatTree.values row7411 = Data.row7411 := rfl
theorem row7411_checked : SortedNatTree.contains row7411 = Data.row7411.contains := by
  rw [SortedNatTree.contains_eq_list row7411 row7411_ordered, row7411_values]
def row7417 : SortedNatTree.Tree := (.node 12973 (.node 10993 (.node 9433 (.node 8017 (.node 7639 .empty .empty) (.node 8263 .empty .empty)) (.node 10789 (.node 9643 .empty .empty) .empty)) (.node 11149 (.node 11119 (.node 11083 .empty .empty) .empty) (.node 12511 (.node 11617 .empty .empty) .empty))) (.node 16747 (.node 14683 (.node 13339 (.node 13267 .empty .empty) (.node 14221 .empty .empty)) (.node 16411 (.node 15187 .empty .empty) .empty)) (.node 17209 (.node 17029 (.node 16879 .empty .empty) .empty) (.node 18583 (.node 18217 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7417_ordered : SortedNatTree.ordered row7417 = true := by decide
set_option maxRecDepth 4096 in
theorem row7417_values : SortedNatTree.values row7417 = Data.row7417 := rfl
theorem row7417_checked : SortedNatTree.contains row7417 = Data.row7417.contains := by
  rw [SortedNatTree.contains_eq_list row7417 row7417_ordered, row7417_values]
def row7433 : SortedNatTree.Tree := (.node 12569 (.node 9677 (.node 8573 (.node 8219 (.node 7559 .empty .empty) .empty) (.node 9467 (.node 8597 .empty .empty) .empty)) (.node 12071 (.node 11057 (.node 9719 .empty .empty) .empty) (.node 12113 (.node 12107 .empty .empty) .empty))) (.node 16187 (.node 15413 (.node 14939 (.node 14669 .empty .empty) .empty) (.node 15959 (.node 15641 .empty .empty) .empty)) (.node 17417 (.node 17189 (.node 16883 .empty .empty) .empty) (.node 17489 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7433_ordered : SortedNatTree.ordered row7433 = true := by decide
set_option maxRecDepth 4096 in
theorem row7433_values : SortedNatTree.values row7433 = Data.row7433 := rfl
theorem row7433_checked : SortedNatTree.contains row7433 = Data.row7433.contains := by
  rw [SortedNatTree.contains_eq_list row7433 row7433_ordered, row7433_values]
def row7451 : SortedNatTree.Tree := (.node 11867 (.node 9677 (.node 8747 (.node 8501 (.node 7853 .empty .empty) .empty) (.node 8819 .empty .empty)) (.node 10607 (.node 9833 .empty .empty) (.node 11273 .empty .empty))) (.node 14537 (.node 12959 (.node 12611 (.node 12281 .empty .empty) .empty) (.node 13001 .empty .empty)) (.node 16139 (.node 15131 .empty .empty) (.node 18251 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7451_ordered : SortedNatTree.ordered row7451 = true := by decide
set_option maxRecDepth 4096 in
theorem row7451_values : SortedNatTree.values row7451 = Data.row7451 := rfl
theorem row7451_checked : SortedNatTree.contains row7451 = Data.row7451.contains := by
  rw [SortedNatTree.contains_eq_list row7451 row7451_ordered, row7451_values]
def row7457 : SortedNatTree.Tree := (.node 14327 (.node 9497 (.node 8513 (.node 8369 (.node 7757 .empty .empty) .empty) (.node 9371 .empty .empty)) (.node 13217 (.node 11261 .empty .empty) (.node 14051 .empty .empty))) (.node 17669 (.node 16889 (.node 16547 (.node 16091 .empty .empty) .empty) (.node 17123 .empty .empty)) (.node 17957 (.node 17939 .empty .empty) (.node 18059 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7457_ordered : SortedNatTree.ordered row7457 = true := by decide
set_option maxRecDepth 4096 in
theorem row7457_values : SortedNatTree.values row7457 = Data.row7457 := rfl
theorem row7457_checked : SortedNatTree.contains row7457 = Data.row7457.contains := by
  rw [SortedNatTree.contains_eq_list row7457 row7457_ordered, row7457_values]
def row7459 : SortedNatTree.Tree := (.node 13009 (.node 9901 (.node 9319 (.node 8521 (.node 7927 .empty .empty) .empty) (.node 9661 (.node 9433 .empty .empty) .empty)) (.node 12073 (.node 10771 (.node 10651 .empty .empty) .empty) (.node 12373 .empty .empty))) (.node 16417 (.node 13711 (.node 13687 (.node 13249 .empty .empty) .empty) (.node 14731 .empty .empty)) (.node 18133 (.node 17911 (.node 17377 .empty .empty) .empty) (.node 18379 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7459_ordered : SortedNatTree.ordered row7459 = true := by decide
set_option maxRecDepth 4096 in
theorem row7459_values : SortedNatTree.values row7459 = Data.row7459 := rfl
theorem row7459_checked : SortedNatTree.contains row7459 = Data.row7459.contains := by
  rw [SortedNatTree.contains_eq_list row7459 row7459_ordered, row7459_values]
def row7477 : SortedNatTree.Tree := (.node 13381 (.node 9829 (.node 9127 (.node 8317 (.node 8191 .empty .empty) (.node 9013 .empty .empty)) (.node 9811 (.node 9391 .empty .empty) .empty)) (.node 11941 (.node 10429 (.node 9907 .empty .empty) (.node 11743 .empty .empty)) (.node 12799 (.node 12253 .empty .empty) .empty))) (.node 15313 (.node 14629 (.node 14197 (.node 13789 .empty .empty) (.node 14533 .empty .empty)) (.node 15217 (.node 15193 .empty .empty) .empty)) (.node 16741 (.node 15973 (.node 15817 .empty .empty) (.node 16603 .empty .empty)) (.node 17293 (.node 17011 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7477_ordered : SortedNatTree.ordered row7477 = true := by decide
set_option maxRecDepth 4096 in
theorem row7477_values : SortedNatTree.values row7477 = Data.row7477 := rfl
theorem row7477_checked : SortedNatTree.contains row7477 = Data.row7477.contains := by
  rw [SortedNatTree.contains_eq_list row7477 row7477_ordered, row7477_values]
def row7481 : SortedNatTree.Tree := (.node 12041 (.node 10037 (.node 9413 (.node 7607 (.node 7541 .empty .empty) (.node 8741 .empty .empty)) (.node 9887 (.node 9551 .empty .empty) .empty)) (.node 11369 (.node 10607 (.node 10499 .empty .empty) .empty) (.node 11783 (.node 11597 .empty .empty) .empty))) (.node 15959 (.node 13649 (.node 13127 (.node 12917 .empty .empty) (.node 13229 .empty .empty)) (.node 14621 (.node 13763 .empty .empty) .empty)) (.node 17477 (.node 17417 (.node 16619 .empty .empty) .empty) (.node 18311 (.node 17729 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7481_ordered : SortedNatTree.ordered row7481 = true := by decide
set_option maxRecDepth 4096 in
theorem row7481_values : SortedNatTree.values row7481 = Data.row7481 := rfl
theorem row7481_checked : SortedNatTree.contains row7481 = Data.row7481.contains := by
  rw [SortedNatTree.contains_eq_list row7481 row7481_ordered, row7481_values]
def row7487 : SortedNatTree.Tree := (.node 13709 (.node 10463 (.node 9923 (.node 9011 (.node 7883 .empty .empty) .empty) (.node 10433 .empty .empty)) (.node 11351 (.node 11273 (.node 11261 .empty .empty) .empty) (.node 12101 .empty .empty))) (.node 15641 (.node 15101 (.node 14723 (.node 14303 .empty .empty) .empty) (.node 15467 .empty .empty)) (.node 17159 (.node 16979 (.node 15797 .empty .empty) .empty) (.node 18041 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7487_ordered : SortedNatTree.ordered row7487 = true := by decide
set_option maxRecDepth 4096 in
theorem row7487_values : SortedNatTree.values row7487 = Data.row7487 := rfl
theorem row7487_checked : SortedNatTree.contains row7487 = Data.row7487.contains := by
  rw [SortedNatTree.contains_eq_list row7487 row7487_ordered, row7487_values]
def row7489 : SortedNatTree.Tree := (.node 11287 (.node 9049 (.node 8737 (.node 8443 (.node 7549 .empty .empty) (.node 8581 .empty .empty)) (.node 9043 (.node 8839 .empty .empty) .empty)) (.node 10039 (.node 9343 (.node 9157 .empty .empty) (.node 9631 .empty .empty)) (.node 10531 (.node 10093 .empty .empty) .empty))) (.node 15439 (.node 14029 (.node 12253 (.node 11887 .empty .empty) (.node 13171 .empty .empty)) (.node 14827 (.node 14143 .empty .empty) .empty)) (.node 17977 (.node 17419 (.node 16699 .empty .empty) (.node 17539 .empty .empty)) (.node 18043 (.node 17989 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7489_ordered : SortedNatTree.ordered row7489 = true := by decide
set_option maxRecDepth 4096 in
theorem row7489_values : SortedNatTree.values row7489 = Data.row7489 := rfl
theorem row7489_checked : SortedNatTree.contains row7489 = Data.row7489.contains := by
  rw [SortedNatTree.contains_eq_list row7489 row7489_ordered, row7489_values]
def row7499 : SortedNatTree.Tree := (.node 13763 (.node 11447 (.node 8369 (.node 7841 (.node 7757 .empty .empty) .empty) (.node 9857 (.node 9059 .empty .empty) .empty)) (.node 11867 (.node 11831 (.node 11717 .empty .empty) .empty) (.node 12377 .empty .empty))) (.node 15629 (.node 14771 (.node 14153 (.node 13859 .empty .empty) .empty) (.node 15077 .empty .empty)) (.node 18149 (.node 17939 (.node 16763 .empty .empty) .empty) (.node 18443 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7499_ordered : SortedNatTree.ordered row7499 = true := by decide
set_option maxRecDepth 4096 in
theorem row7499_values : SortedNatTree.values row7499 = Data.row7499 := rfl
theorem row7499_checked : SortedNatTree.contains row7499 = Data.row7499.contains := by
  rw [SortedNatTree.contains_eq_list row7499 row7499_ordered, row7499_values]
def row7507 : SortedNatTree.Tree := (.node 11491 (.node 9643 (.node 8893 (.node 8467 (.node 8431 .empty .empty) .empty) (.node 9187 .empty .empty)) (.node 11287 (.node 9859 .empty .empty) (.node 11443 .empty .empty))) (.node 16567 (.node 12577 (.node 11839 .empty .empty) (.node 13417 .empty .empty)) (.node 17971 (.node 17299 .empty .empty) (.node 18121 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7507_ordered : SortedNatTree.ordered row7507 = true := by decide
set_option maxRecDepth 4096 in
theorem row7507_values : SortedNatTree.values row7507 = Data.row7507 := rfl
theorem row7507_checked : SortedNatTree.contains row7507 = Data.row7507.contains := by
  rw [SortedNatTree.contains_eq_list row7507 row7507_ordered, row7507_values]
def row7517 : SortedNatTree.Tree := (.node 12659 (.node 11057 (.node 8387 (.node 7829 (.node 7727 .empty .empty) .empty) (.node 9551 (.node 8861 .empty .empty) .empty)) (.node 12113 (.node 11981 (.node 11399 .empty .empty) .empty) (.node 12527 .empty .empty))) (.node 14969 (.node 13367 (.node 13121 (.node 12743 .empty .empty) .empty) (.node 14591 .empty .empty)) (.node 16607 (.node 16187 (.node 16103 .empty .empty) .empty) (.node 17669 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7517_ordered : SortedNatTree.ordered row7517 = true := by decide
set_option maxRecDepth 4096 in
theorem row7517_values : SortedNatTree.values row7517 = Data.row7517 := rfl
theorem row7517_checked : SortedNatTree.contains row7517 = Data.row7517.contains := by
  rw [SortedNatTree.contains_eq_list row7517 row7517_ordered, row7517_values]
def row7523 : SortedNatTree.Tree := (.node 12227 (.node 10463 (.node 8783 (.node 7727 (.node 7541 .empty .empty) .empty) (.node 10331 (.node 9431 .empty .empty) .empty)) (.node 11867 (.node 10733 (.node 10607 .empty .empty) .empty) (.node 12161 .empty .empty))) (.node 16553 (.node 13229 (.node 12641 (.node 12479 .empty .empty) .empty) (.node 15791 (.node 14657 .empty .empty) .empty)) (.node 18401 (.node 17921 (.node 16619 .empty .empty) .empty) (.node 18461 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7523_ordered : SortedNatTree.ordered row7523 = true := by decide
set_option maxRecDepth 4096 in
theorem row7523_values : SortedNatTree.values row7523 = Data.row7523 := rfl
theorem row7523_checked : SortedNatTree.contains row7523 = Data.row7523.contains := by
  rw [SortedNatTree.contains_eq_list row7523 row7523_ordered, row7523_values]
def row7529 : SortedNatTree.Tree := (.node 12101 (.node 9227 (.node 8753 (.node 7877 .empty .empty) (.node 8999 .empty .empty)) (.node 9941 (.node 9479 .empty .empty) (.node 10253 .empty .empty))) (.node 16607 (.node 14831 (.node 14549 .empty .empty) (.node 15017 .empty .empty)) (.node 17183 (.node 17027 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7529_ordered : SortedNatTree.ordered row7529 = true := by decide
set_option maxRecDepth 4096 in
theorem row7529_values : SortedNatTree.values row7529 = Data.row7529 := rfl
theorem row7529_checked : SortedNatTree.contains row7529 = Data.row7529.contains := by
  rw [SortedNatTree.contains_eq_list row7529 row7529_ordered, row7529_values]
def row7537 : SortedNatTree.Tree := (.node 11821 (.node 9949 (.node 8419 (.node 8179 (.node 8101 .empty .empty) .empty) (.node 9781 (.node 8713 .empty .empty) .empty)) (.node 10429 (.node 10303 (.node 10267 .empty .empty) .empty) (.node 11689 (.node 10729 .empty .empty) .empty))) (.node 15667 (.node 14869 (.node 13327 (.node 12097 .empty .empty) .empty) (.node 15661 (.node 15241 .empty .empty) .empty)) (.node 17299 (.node 16273 (.node 16057 .empty .empty) .empty) (.node 17989 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7537_ordered : SortedNatTree.ordered row7537 = true := by decide
set_option maxRecDepth 4096 in
theorem row7537_values : SortedNatTree.values row7537 = Data.row7537 := rfl
theorem row7537_checked : SortedNatTree.contains row7537 = Data.row7537.contains := by
  rw [SortedNatTree.contains_eq_list row7537 row7537_ordered, row7537_values]
def row7541 : SortedNatTree.Tree := (.node 9941 (.node 8699 (.node 8111 (.node 7883 .empty .empty) (.node 8423 .empty .empty)) (.node 9011 (.node 8747 .empty .empty) (.node 9857 .empty .empty))) (.node 15497 (.node 11471 (.node 10457 .empty .empty) (.node 15443 .empty .empty)) (.node 17981 (.node 17393 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7541_ordered : SortedNatTree.ordered row7541 = true := by decide
set_option maxRecDepth 4096 in
theorem row7541_values : SortedNatTree.values row7541 = Data.row7541 := rfl
theorem row7541_checked : SortedNatTree.contains row7541 = Data.row7541.contains := by
  rw [SortedNatTree.contains_eq_list row7541 row7541_ordered, row7541_values]
def row7547 : SortedNatTree.Tree := (.node 12377 (.node 9923 (.node 9059 (.node 8093 .empty .empty) (.node 9857 .empty .empty)) (.node 12113 (.node 10487 .empty .empty) .empty)) (.node 13499 (.node 12743 (.node 12611 .empty .empty) .empty) (.node 14669 (.node 13721 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7547_ordered : SortedNatTree.ordered row7547 = true := by decide
set_option maxRecDepth 4096 in
theorem row7547_values : SortedNatTree.values row7547 = Data.row7547 := rfl
theorem row7547_checked : SortedNatTree.contains row7547 = Data.row7547.contains := by
  rw [SortedNatTree.contains_eq_list row7547 row7547_ordered, row7547_values]
def row7549 : SortedNatTree.Tree := (.node 11923 (.node 9349 (.node 8893 (.node 8209 (.node 8089 .empty .empty) .empty) (.node 9049 (.node 8923 .empty .empty) .empty)) (.node 9967 (.node 9859 (.node 9511 .empty .empty) .empty) (.node 10477 .empty .empty))) (.node 14653 (.node 12409 (.node 12289 (.node 12157 .empty .empty) .empty) (.node 12799 .empty .empty)) (.node 18181 (.node 17467 (.node 14851 .empty .empty) .empty) (.node 18427 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7549_ordered : SortedNatTree.ordered row7549 = true := by decide
set_option maxRecDepth 4096 in
theorem row7549_values : SortedNatTree.values row7549 = Data.row7549 := rfl
theorem row7549_checked : SortedNatTree.contains row7549 = Data.row7549.contains := by
  rw [SortedNatTree.contains_eq_list row7549 row7549_ordered, row7549_values]
def row7559 : SortedNatTree.Tree := (.node 14153 (.node 10667 (.node 8849 (.node 8597 (.node 8111 .empty .empty) .empty) (.node 9281 .empty .empty)) (.node 13523 (.node 11159 .empty .empty) (.node 14057 .empty .empty))) (.node 15581 (.node 14813 (.node 14657 (.node 14321 .empty .empty) .empty) (.node 15527 .empty .empty)) (.node 17783 (.node 16979 .empty .empty) (.node 17837 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7559_ordered : SortedNatTree.ordered row7559 = true := by decide
set_option maxRecDepth 4096 in
theorem row7559_values : SortedNatTree.values row7559 = Data.row7559 := rfl
theorem row7559_checked : SortedNatTree.contains row7559 = Data.row7559.contains := by
  rw [SortedNatTree.contains_eq_list row7559 row7559_ordered, row7559_values]
def row7561 : SortedNatTree.Tree := (.node 12919 (.node 9127 (.node 8821 (.node 7963 (.node 7723 .empty .empty) .empty) (.node 8887 .empty .empty)) (.node 9463 (.node 9433 (.node 9337 .empty .empty) .empty) (.node 10891 .empty .empty))) (.node 17599 (.node 16519 (.node 16069 (.node 15667 .empty .empty) .empty) (.node 16657 .empty .empty)) (.node 18379 (.node 18229 .empty .empty) (.node 18397 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7561_ordered : SortedNatTree.ordered row7561 = true := by decide
set_option maxRecDepth 4096 in
theorem row7561_values : SortedNatTree.values row7561 = Data.row7561 := rfl
theorem row7561_checked : SortedNatTree.contains row7561 = Data.row7561.contains := by
  rw [SortedNatTree.contains_eq_list row7561 row7561_ordered, row7561_values]
def row7573 : SortedNatTree.Tree := (.node 11719 (.node 9133 (.node 8329 (.node 7621 (.node 7603 .empty .empty) (.node 8263 .empty .empty)) (.node 8539 (.node 8419 .empty .empty) .empty)) (.node 10891 (.node 10651 (.node 9787 .empty .empty) .empty) (.node 11383 (.node 11161 .empty .empty) .empty))) (.node 15217 (.node 13633 (.node 12979 (.node 12163 .empty .empty) (.node 13537 .empty .empty)) (.node 14563 (.node 13831 .empty .empty) .empty)) (.node 16141 (.node 15991 (.node 15739 .empty .empty) .empty) (.node 18427 (.node 17203 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7573_ordered : SortedNatTree.ordered row7573 = true := by decide
set_option maxRecDepth 4096 in
theorem row7573_values : SortedNatTree.values row7573 = Data.row7573 := rfl
theorem row7573_checked : SortedNatTree.contains row7573 = Data.row7573.contains := by
  rw [SortedNatTree.contains_eq_list row7573 row7573_ordered, row7573_values]
def row7577 : SortedNatTree.Tree := (.node 12653 (.node 9461 (.node 8693 (.node 8543 (.node 8069 .empty .empty) (.node 8609 .empty .empty)) (.node 9413 (.node 8699 .empty .empty) .empty)) (.node 11177 (.node 10433 (.node 10259 .empty .empty) .empty) (.node 12227 (.node 12119 .empty .empty) .empty))) (.node 15971 (.node 14831 (.node 14303 (.node 12911 .empty .empty) (.node 14411 .empty .empty)) (.node 15731 (.node 15641 .empty .empty) .empty)) (.node 16811 (.node 16361 (.node 16103 .empty .empty) .empty) (.node 17729 (.node 17027 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7577_ordered : SortedNatTree.ordered row7577 = true := by decide
set_option maxRecDepth 4096 in
theorem row7577_values : SortedNatTree.values row7577 = Data.row7577 := rfl
theorem row7577_checked : SortedNatTree.contains row7577 = Data.row7577.contains := by
  rw [SortedNatTree.contains_eq_list row7577 row7577_ordered, row7577_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
