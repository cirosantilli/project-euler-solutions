import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row5861 : SortedNatTree.Tree := (.node 12941 (.node 9059 (.node 6827 (.node 6143 (.node 5987 .empty .empty) .empty) (.node 8171 (.node 7043 .empty .empty) .empty)) (.node 10883 (.node 10169 (.node 9221 .empty .empty) .empty) (.node 11681 (.node 11621 .empty .empty) .empty))) (.node 16451 (.node 14327 (.node 13679 (.node 13613 .empty .empty) .empty) (.node 15329 (.node 14537 .empty .empty) .empty)) (.node 19031 (.node 18353 (.node 17183 .empty .empty) .empty) (.node 19991 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5861_ordered : SortedNatTree.ordered row5861 = true := by decide
set_option maxRecDepth 4096 in
theorem row5861_values : SortedNatTree.values row5861 = Data.row5861 := rfl
theorem row5861_checked : SortedNatTree.contains row5861 = Data.row5861.contains := by
  rw [SortedNatTree.contains_eq_list row5861 row5861_ordered, row5861_values]
def row5867 : SortedNatTree.Tree := (.node 14657 (.node 10781 (.node 9281 (.node 7001 (.node 6329 .empty .empty) .empty) (.node 10067 (.node 10037 .empty .empty) .empty)) (.node 11897 (.node 11813 (.node 11159 .empty .empty) .empty) (.node 13451 .empty .empty))) (.node 18269 (.node 16673 (.node 15887 (.node 15749 .empty .empty) .empty) (.node 17609 .empty .empty)) (.node 19853 (.node 19751 (.node 18329 .empty .empty) .empty) (.node 19979 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5867_ordered : SortedNatTree.ordered row5867 = true := by decide
set_option maxRecDepth 4096 in
theorem row5867_values : SortedNatTree.values row5867 = Data.row5867 := rfl
theorem row5867_checked : SortedNatTree.contains row5867 = Data.row5867.contains := by
  rw [SortedNatTree.contains_eq_list row5867 row5867_ordered, row5867_values]
def row5869 : SortedNatTree.Tree := (.node 10867 (.node 8089 (.node 6829 (.node 6163 (.node 6067 .empty .empty) (.node 6709 .empty .empty)) (.node 7069 (.node 6961 .empty .empty) .empty)) (.node 9547 (.node 9439 (.node 8209 .empty .empty) .empty) (.node 10657 (.node 9967 .empty .empty) .empty))) (.node 15313 (.node 12577 (.node 11239 (.node 10993 .empty .empty) (.node 11839 .empty .empty)) (.node 13597 (.node 12967 .empty .empty) .empty)) (.node 19483 (.node 19183 (.node 15937 .empty .empty) .empty) (.node 19687 (.node 19609 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5869_ordered : SortedNatTree.ordered row5869 = true := by decide
set_option maxRecDepth 4096 in
theorem row5869_values : SortedNatTree.values row5869 = Data.row5869 := rfl
theorem row5869_checked : SortedNatTree.contains row5869 = Data.row5869.contains := by
  rw [SortedNatTree.contains_eq_list row5869 row5869_ordered, row5869_values]
def row5879 : SortedNatTree.Tree := (.node 15137 (.node 9677 (.node 7127 (.node 6779 (.node 6263 .empty .empty) .empty) (.node 8273 .empty .empty)) (.node 12239 (.node 10667 (.node 9887 .empty .empty) .empty) (.node 12251 .empty .empty))) (.node 16883 (.node 16073 (.node 15569 (.node 15299 .empty .empty) .empty) (.node 16673 .empty .empty)) (.node 17909 (.node 17021 .empty .empty) (.node 20051 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5879_ordered : SortedNatTree.ordered row5879 = true := by decide
set_option maxRecDepth 4096 in
theorem row5879_values : SortedNatTree.values row5879 = Data.row5879 := rfl
theorem row5879_checked : SortedNatTree.contains row5879 = Data.row5879.contains := by
  rw [SortedNatTree.contains_eq_list row5879 row5879_ordered, row5879_values]
def row5881 : SortedNatTree.Tree := (.node 13177 (.node 10867 (.node 8779 (.node 8269 (.node 7723 .empty .empty) .empty) (.node 10429 (.node 9547 .empty .empty) .empty)) (.node 11971 (.node 11497 (.node 11299 .empty .empty) .empty) (.node 12211 .empty .empty))) (.node 17293 (.node 16057 (.node 15991 (.node 14533 .empty .empty) .empty) (.node 16657 (.node 16453 .empty .empty) .empty)) (.node 19531 (.node 19417 (.node 18133 .empty .empty) .empty) (.node 19603 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5881_ordered : SortedNatTree.ordered row5881 = true := by decide
set_option maxRecDepth 4096 in
theorem row5881_values : SortedNatTree.values row5881 = Data.row5881 := rfl
theorem row5881_checked : SortedNatTree.contains row5881 = Data.row5881.contains := by
  rw [SortedNatTree.contains_eq_list row5881 row5881_ordered, row5881_values]
def row5897 : SortedNatTree.Tree := (.node 11519 (.node 9281 (.node 7019 (.node 6143 (.node 6131 .empty .empty) (.node 6977 .empty .empty)) (.node 8861 (.node 8537 .empty .empty) (.node 8867 .empty .empty))) (.node 10937 (.node 9803 (.node 9467 .empty .empty) (.node 9923 .empty .empty)) (.node 11279 (.node 11261 .empty .empty) .empty))) (.node 15461 (.node 13109 (.node 12281 (.node 12197 .empty .empty) (.node 12653 .empty .empty)) (.node 14627 (.node 13829 .empty .empty) .empty)) (.node 17747 (.node 16823 (.node 15647 .empty .empty) (.node 17327 .empty .empty)) (.node 19841 (.node 19301 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5897_ordered : SortedNatTree.ordered row5897 = true := by decide
set_option maxRecDepth 4096 in
theorem row5897_values : SortedNatTree.values row5897 = Data.row5897 := rfl
theorem row5897_checked : SortedNatTree.contains row5897 = Data.row5897.contains := by
  rw [SortedNatTree.contains_eq_list row5897 row5897_ordered, row5897_values]
def row5903 : SortedNatTree.Tree := (.node 10529 (.node 7457 (.node 7001 (.node 6311 (.node 6143 .empty .empty) (.node 6827 .empty .empty)) (.node 7211 (.node 7109 .empty .empty) .empty)) (.node 8009 (.node 7877 (.node 7523 .empty .empty) .empty) (.node 9293 (.node 9059 .empty .empty) .empty))) (.node 14537 (.node 12941 (.node 11633 (.node 11489 .empty .empty) .empty) (.node 14009 (.node 13553 .empty .empty) .empty)) (.node 17921 (.node 17579 (.node 14867 .empty .empty) .empty) (.node 20021 (.node 18593 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5903_ordered : SortedNatTree.ordered row5903 = true := by decide
set_option maxRecDepth 4096 in
theorem row5903_values : SortedNatTree.values row5903 = Data.row5903 := rfl
theorem row5903_checked : SortedNatTree.contains row5903 = Data.row5903.contains := by
  rw [SortedNatTree.contains_eq_list row5903 row5903_ordered, row5903_values]
def row5923 : SortedNatTree.Tree := (.node 14407 (.node 8581 (.node 7177 (.node 6427 (.node 6421 .empty .empty) (.node 6871 .empty .empty)) (.node 7963 (.node 7417 .empty .empty) (.node 8467 .empty .empty))) (.node 9013 (.node 8689 (.node 8623 .empty .empty) (.node 8929 .empty .empty)) (.node 12757 (.node 10903 .empty .empty) .empty))) (.node 16993 (.node 16183 (.node 15289 (.node 14851 .empty .empty) (.node 16111 .empty .empty)) (.node 16927 (.node 16657 .empty .empty) .empty)) (.node 19447 (.node 18553 (.node 18367 .empty .empty) (.node 19249 .empty .empty)) (.node 19687 (.node 19543 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5923_ordered : SortedNatTree.ordered row5923 = true := by decide
set_option maxRecDepth 4096 in
theorem row5923_values : SortedNatTree.values row5923 = Data.row5923 := rfl
theorem row5923_checked : SortedNatTree.contains row5923 = Data.row5923.contains := by
  rw [SortedNatTree.contains_eq_list row5923 row5923_ordered, row5923_values]
def row5927 : SortedNatTree.Tree := (.node 13613 (.node 9851 (.node 8237 (.node 7499 (.node 5939 .empty .empty) (.node 7583 .empty .empty)) (.node 9137 (.node 8861 .empty .empty) (.node 9749 .empty .empty))) (.node 11789 (.node 11369 (.node 10247 .empty .empty) (.node 11489 .empty .empty)) (.node 13001 (.node 12413 .empty .empty) (.node 13523 .empty .empty)))) (.node 15737 (.node 14051 (.node 13907 (.node 13883 .empty .empty) (.node 14009 .empty .empty)) (.node 15287 (.node 14303 .empty .empty) (.node 15329 .empty .empty))) (.node 17207 (.node 16253 (.node 16193 .empty .empty) (.node 16889 .empty .empty)) (.node 19157 (.node 18773 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5927_ordered : SortedNatTree.ordered row5927 = true := by decide
set_option maxRecDepth 4096 in
theorem row5927_values : SortedNatTree.values row5927 = Data.row5927 := rfl
theorem row5927_checked : SortedNatTree.contains row5927 = Data.row5927.contains := by
  rw [SortedNatTree.contains_eq_list row5927 row5927_ordered, row5927_values]
def row5939 : SortedNatTree.Tree := (.node 11447 (.node 7817 (.node 6269 (.node 6101 (.node 6053 .empty .empty) (.node 6173 .empty .empty)) (.node 7529 (.node 6551 .empty .empty) .empty)) (.node 8861 (.node 8501 (.node 8243 .empty .empty) .empty) (.node 10061 (.node 9497 .empty .empty) .empty))) (.node 14723 (.node 12899 (.node 11807 (.node 11489 .empty .empty) .empty) (.node 13829 (.node 13757 .empty .empty) .empty)) (.node 17609 (.node 17033 (.node 16139 .empty .empty) .empty) (.node 19763 (.node 19121 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5939_ordered : SortedNatTree.ordered row5939 = true := by decide
set_option maxRecDepth 4096 in
theorem row5939_values : SortedNatTree.values row5939 = Data.row5939 := rfl
theorem row5939_checked : SortedNatTree.contains row5939 = Data.row5939.contains := by
  rw [SortedNatTree.contains_eq_list row5939 row5939_ordered, row5939_values]
def row5953 : SortedNatTree.Tree := (.node 13879 (.node 11863 (.node 8923 (.node 6277 (.node 6199 .empty .empty) (.node 7039 .empty .empty)) (.node 10567 (.node 9679 .empty .empty) .empty)) (.node 13093 (.node 12457 (.node 12379 .empty .empty) (.node 12973 .empty .empty)) (.node 13309 (.node 13171 .empty .empty) .empty))) (.node 17167 (.node 15241 (.node 14563 (.node 14107 .empty .empty) (.node 14851 .empty .empty)) (.node 16747 (.node 16369 .empty .empty) .empty)) (.node 18211 (.node 17659 (.node 17293 .empty .empty) (.node 17881 .empty .empty)) (.node 19819 (.node 18301 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5953_ordered : SortedNatTree.ordered row5953 = true := by decide
set_option maxRecDepth 4096 in
theorem row5953_values : SortedNatTree.values row5953 = Data.row5953 := rfl
theorem row5953_checked : SortedNatTree.contains row5953 = Data.row5953.contains := by
  rw [SortedNatTree.contains_eq_list row5953 row5953_ordered, row5953_values]
def row5981 : SortedNatTree.Tree := (.node 13463 (.node 9323 (.node 8273 (.node 8081 (.node 7283 .empty .empty) (.node 8237 .empty .empty)) (.node 8819 (.node 8543 .empty .empty) .empty)) (.node 11933 (.node 9497 (.node 9341 .empty .empty) (.node 11471 .empty .empty)) (.node 12899 (.node 12329 .empty .empty) .empty))) (.node 16193 (.node 14669 (.node 14051 (.node 13913 .empty .empty) (.node 14549 .empty .empty)) (.node 16061 (.node 14969 .empty .empty) .empty)) (.node 17903 (.node 17033 (.node 16361 .empty .empty) .empty) (.node 19421 (.node 18959 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5981_ordered : SortedNatTree.ordered row5981 = true := by decide
set_option maxRecDepth 4096 in
theorem row5981_values : SortedNatTree.values row5981 = Data.row5981 := rfl
theorem row5981_checked : SortedNatTree.contains row5981 = Data.row5981.contains := by
  rw [SortedNatTree.contains_eq_list row5981 row5981_ordered, row5981_values]
def row5987 : SortedNatTree.Tree := (.node 10613 (.node 8831 (.node 7817 (.node 6491 (.node 6389 .empty .empty) .empty) (.node 8513 (.node 8363 .empty .empty) .empty)) (.node 10193 (.node 10163 (.node 9059 .empty .empty) .empty) (.node 10253 .empty .empty))) (.node 14621 (.node 12497 (.node 11447 (.node 11213 .empty .empty) .empty) (.node 14489 (.node 14321 .empty .empty) .empty)) (.node 18269 (.node 17099 (.node 15629 .empty .empty) .empty) (.node 19259 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5987_ordered : SortedNatTree.ordered row5987 = true := by decide
set_option maxRecDepth 4096 in
theorem row5987_values : SortedNatTree.values row5987 = Data.row5987 := rfl
theorem row5987_checked : SortedNatTree.contains row5987 = Data.row5987.contains := by
  rw [SortedNatTree.contains_eq_list row5987 row5987_ordered, row5987_values]
def row6007 : SortedNatTree.Tree := (.node 11299 (.node 8443 (.node 7333 (.node 6091 (.node 6073 .empty .empty) (.node 7057 .empty .empty)) (.node 7681 (.node 7489 .empty .empty) (.node 7933 .empty .empty))) (.node 10597 (.node 8761 (.node 8581 .empty .empty) (.node 9337 .empty .empty)) (.node 10837 (.node 10723 .empty .empty) (.node 11071 .empty .empty)))) (.node 14503 (.node 13297 (.node 12589 (.node 11731 .empty .empty) (.node 12613 .empty .empty)) (.node 14281 (.node 13831 .empty .empty) (.node 14461 .empty .empty))) (.node 18661 (.node 15187 (.node 14923 .empty .empty) (.node 15991 .empty .empty)) (.node 19501 (.node 18973 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6007_ordered : SortedNatTree.ordered row6007 = true := by decide
set_option maxRecDepth 4096 in
theorem row6007_values : SortedNatTree.values row6007 = Data.row6007 := rfl
theorem row6007_checked : SortedNatTree.contains row6007 = Data.row6007.contains := by
  rw [SortedNatTree.contains_eq_list row6007 row6007_ordered, row6007_values]
def row6011 : SortedNatTree.Tree := (.node 14741 (.node 8741 (.node 7451 (.node 7307 (.node 7019 .empty .empty) .empty) (.node 7727 .empty .empty)) (.node 13859 (.node 13229 (.node 11087 .empty .empty) .empty) (.node 14051 .empty .empty))) (.node 18059 (.node 16871 (.node 16193 (.node 14783 .empty .empty) .empty) (.node 17183 .empty .empty)) (.node 18269 (.node 18191 .empty .empty) (.node 18803 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6011_ordered : SortedNatTree.ordered row6011 = true := by decide
set_option maxRecDepth 4096 in
theorem row6011_values : SortedNatTree.values row6011 = Data.row6011 := rfl
theorem row6011_checked : SortedNatTree.contains row6011 = Data.row6011.contains := by
  rw [SortedNatTree.contains_eq_list row6011 row6011_ordered, row6011_values]
def row6029 : SortedNatTree.Tree := (.node 12413 (.node 9203 (.node 7607 (.node 6491 (.node 6389 (.node 6221 .empty .empty) .empty) (.node 7529 .empty .empty)) (.node 8423 (.node 7649 .empty .empty) (.node 8783 .empty .empty))) (.node 9923 (.node 9539 (.node 9419 .empty .empty) (.node 9623 .empty .empty)) (.node 11447 (.node 11003 .empty .empty) (.node 12269 .empty .empty)))) (.node 16661 (.node 14369 (.node 13901 (.node 13613 .empty .empty) (.node 14243 .empty .empty)) (.node 14867 (.node 14771 .empty .empty) (.node 15233 .empty .empty))) (.node 18713 (.node 18269 (.node 16673 .empty .empty) (.node 18401 .empty .empty)) (.node 19973 (.node 19301 .empty .empty) (.node 19979 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6029_ordered : SortedNatTree.ordered row6029 = true := by decide
set_option maxRecDepth 4096 in
theorem row6029_values : SortedNatTree.values row6029 = Data.row6029 := rfl
theorem row6029_checked : SortedNatTree.contains row6029 = Data.row6029.contains := by
  rw [SortedNatTree.contains_eq_list row6029 row6029_ordered, row6029_values]
def row6037 : SortedNatTree.Tree := (.node 13417 (.node 9643 (.node 8719 (.node 8209 (.node 6991 .empty .empty) .empty) (.node 9433 (.node 9013 .empty .empty) .empty)) (.node 13093 (.node 12457 (.node 10099 .empty .empty) .empty) (.node 13381 (.node 13291 .empty .empty) .empty))) (.node 17383 (.node 16453 (.node 15601 (.node 14887 .empty .empty) .empty) (.node 17203 (.node 16741 .empty .empty) .empty)) (.node 19423 (.node 18169 (.node 17827 .empty .empty) .empty) (.node 19891 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6037_ordered : SortedNatTree.ordered row6037 = true := by decide
set_option maxRecDepth 4096 in
theorem row6037_values : SortedNatTree.values row6037 = Data.row6037 := rfl
theorem row6037_checked : SortedNatTree.contains row6037 = Data.row6037.contains := by
  rw [SortedNatTree.contains_eq_list row6037 row6037_ordered, row6037_values]
def row6043 : SortedNatTree.Tree := (.node 15139 (.node 10513 (.node 6619 (.node 6553 (.node 6451 .empty .empty) .empty) (.node 8461 (.node 7219 .empty .empty) .empty)) (.node 13159 (.node 13033 (.node 12517 .empty .empty) .empty) (.node 14281 .empty .empty))) (.node 18367 (.node 16183 (.node 15493 (.node 15307 .empty .empty) .empty) (.node 17011 (.node 16963 .empty .empty) .empty)) (.node 19447 (.node 19381 (.node 19249 .empty .empty) .empty) (.node 19867 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6043_ordered : SortedNatTree.ordered row6043 = true := by decide
set_option maxRecDepth 4096 in
theorem row6043_values : SortedNatTree.values row6043 = Data.row6043 := rfl
theorem row6043_checked : SortedNatTree.contains row6043 = Data.row6043.contains := by
  rw [SortedNatTree.contains_eq_list row6043 row6043_ordered, row6043_values]
def row6047 : SortedNatTree.Tree := (.node 11243 (.node 8447 (.node 6719 (.node 6359 (.node 6203 .empty .empty) (.node 6569 .empty .empty)) (.node 7643 (.node 6737 .empty .empty) (.node 8069 .empty .empty))) (.node 9467 (.node 9227 (.node 8963 .empty .empty) (.node 9311 .empty .empty)) (.node 9803 (.node 9623 .empty .empty) .empty))) (.node 14969 (.node 12503 (.node 11717 (.node 11519 .empty .empty) (.node 12401 .empty .empty)) (.node 14081 (.node 13997 .empty .empty) (.node 14543 .empty .empty))) (.node 19463 (.node 17783 (.node 17681 .empty .empty) (.node 18341 .empty .empty)) (.node 19961 (.node 19727 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6047_ordered : SortedNatTree.ordered row6047 = true := by decide
set_option maxRecDepth 4096 in
theorem row6047_values : SortedNatTree.values row6047 = Data.row6047 := rfl
theorem row6047_checked : SortedNatTree.contains row6047 = Data.row6047.contains := by
  rw [SortedNatTree.contains_eq_list row6047 row6047_ordered, row6047_values]
def row6053 : SortedNatTree.Tree := (.node 13469 (.node 10169 (.node 9137 (.node 6977 (.node 6173 .empty .empty) (.node 8171 .empty .empty)) (.node 9767 (.node 9467 .empty .empty) .empty)) (.node 12143 (.node 11321 (.node 10457 .empty .empty) (.node 11489 .empty .empty)) (.node 13421 (.node 12503 .empty .empty) .empty))) (.node 16007 (.node 15329 (.node 14867 (.node 14243 .empty .empty) (.node 15149 .empty .empty)) (.node 15959 (.node 15791 .empty .empty) .empty)) (.node 17807 (.node 17393 (.node 16703 .empty .empty) (.node 17477 .empty .empty)) (.node 19739 (.node 19073 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6053_ordered : SortedNatTree.ordered row6053 = true := by decide
set_option maxRecDepth 4096 in
theorem row6053_values : SortedNatTree.values row6053 = Data.row6053 := rfl
theorem row6053_checked : SortedNatTree.contains row6053 = Data.row6053.contains := by
  rw [SortedNatTree.contains_eq_list row6053 row6053_ordered, row6053_values]
def row6067 : SortedNatTree.Tree := (.node 15289 (.node 9739 (.node 9067 (.node 7993 (.node 7699 .empty .empty) (.node 8221 .empty .empty)) (.node 9187 (.node 9127 .empty .empty) .empty)) (.node 12697 (.node 12409 (.node 12109 .empty .empty) .empty) (.node 14929 (.node 12703 .empty .empty) .empty))) (.node 18289 (.node 17359 (.node 16963 (.node 16063 .empty .empty) (.node 17257 .empty .empty)) (.node 18061 (.node 17491 .empty .empty) .empty)) (.node 19441 (.node 18979 (.node 18787 .empty .empty) .empty) (.node 19801 (.node 19753 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6067_ordered : SortedNatTree.ordered row6067 = true := by decide
set_option maxRecDepth 4096 in
theorem row6067_values : SortedNatTree.values row6067 = Data.row6067 := rfl
theorem row6067_checked : SortedNatTree.contains row6067 = Data.row6067.contains := by
  rw [SortedNatTree.contains_eq_list row6067 row6067_ordered, row6067_values]
def row6073 : SortedNatTree.Tree := (.node 11161 (.node 8647 (.node 7681 (.node 6367 (.node 6199 .empty .empty) (.node 7219 .empty .empty)) (.node 7759 (.node 7723 .empty .empty) (.node 8563 .empty .empty))) (.node 9391 (.node 8761 (.node 8677 .empty .empty) (.node 9187 .empty .empty)) (.node 9907 (.node 9769 .empty .empty) .empty))) (.node 16249 (.node 15073 (.node 12763 (.node 12073 .empty .empty) (.node 14731 .empty .empty)) (.node 15349 (.node 15307 .empty .empty) .empty)) (.node 18133 (.node 17749 (.node 16417 .empty .empty) (.node 17959 .empty .empty)) (.node 19387 (.node 18433 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6073_ordered : SortedNatTree.ordered row6073 = true := by decide
set_option maxRecDepth 4096 in
theorem row6073_values : SortedNatTree.values row6073 = Data.row6073 := rfl
theorem row6073_checked : SortedNatTree.contains row6073 = Data.row6073.contains := by
  rw [SortedNatTree.contains_eq_list row6073 row6073_ordered, row6073_values]
def row6079 : SortedNatTree.Tree := (.node 12919 (.node 9733 (.node 7753 (.node 7621 (.node 6451 .empty .empty) .empty) (.node 8923 (.node 7927 .empty .empty) .empty)) (.node 11527 (.node 11287 (.node 10987 .empty .empty) .empty) (.node 11743 .empty .empty))) (.node 17077 (.node 15727 (.node 13309 (.node 13099 .empty .empty) .empty) (.node 17047 .empty .empty)) (.node 18379 (.node 17707 (.node 17341 .empty .empty) .empty) (.node 19861 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6079_ordered : SortedNatTree.ordered row6079 = true := by decide
set_option maxRecDepth 4096 in
theorem row6079_values : SortedNatTree.values row6079 = Data.row6079 := rfl
theorem row6079_checked : SortedNatTree.contains row6079 = Data.row6079.contains := by
  rw [SortedNatTree.contains_eq_list row6079 row6079_ordered, row6079_values]
def row6089 : SortedNatTree.Tree := (.node 11717 (.node 8861 (.node 7433 (.node 6359 (.node 6131 .empty .empty) (.node 7127 .empty .empty)) (.node 8111 (.node 7649 .empty .empty) (.node 8693 .empty .empty))) (.node 9857 (.node 9551 (.node 9467 .empty .empty) (.node 9749 .empty .empty)) (.node 10343 (.node 10259 .empty .empty) (.node 10853 .empty .empty)))) (.node 17021 (.node 14549 (.node 12569 (.node 11789 .empty .empty) (.node 14519 .empty .empty)) (.node 15467 (.node 14897 .empty .empty) (.node 16001 .empty .empty))) (.node 18059 (.node 17417 (.node 17291 .empty .empty) (.node 17807 .empty .empty)) (.node 19433 (.node 19037 .empty .empty) (.node 19463 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6089_ordered : SortedNatTree.ordered row6089 = true := by decide
set_option maxRecDepth 4096 in
theorem row6089_values : SortedNatTree.values row6089 = Data.row6089 := rfl
theorem row6089_checked : SortedNatTree.contains row6089 = Data.row6089.contains := by
  rw [SortedNatTree.contains_eq_list row6089 row6089_ordered, row6089_values]
def row6091 : SortedNatTree.Tree := (.node 14557 (.node 13411 (.node 10861 (.node 9043 (.node 8311 .empty .empty) .empty) (.node 13033 .empty .empty)) (.node 13633 (.node 13591 .empty .empty) (.node 13693 .empty .empty))) (.node 17959 (.node 17749 (.node 17107 (.node 15919 .empty .empty) .empty) (.node 17863 .empty .empty)) (.node 19447 (.node 18199 .empty .empty) (.node 19501 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6091_ordered : SortedNatTree.ordered row6091 = true := by decide
set_option maxRecDepth 4096 in
theorem row6091_values : SortedNatTree.values row6091 = Data.row6091 := rfl
theorem row6091_checked : SortedNatTree.contains row6091 = Data.row6091.contains := by
  rw [SortedNatTree.contains_eq_list row6091 row6091_ordered, row6091_values]
def row6101 : SortedNatTree.Tree := (.node 11471 (.node 7883 (.node 7487 (.node 7211 (.node 6947 .empty .empty) .empty) (.node 7877 (.node 7589 .empty .empty) .empty)) (.node 8753 (.node 8123 (.node 8093 .empty .empty) .empty) (.node 9587 (.node 9293 .empty .empty) .empty))) (.node 14957 (.node 14081 (.node 12161 (.node 11867 .empty .empty) .empty) (.node 14759 (.node 14369 .empty .empty) .empty)) (.node 18413 (.node 16433 (.node 15053 .empty .empty) .empty) (.node 19031 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6101_ordered : SortedNatTree.ordered row6101 = true := by decide
set_option maxRecDepth 4096 in
theorem row6101_values : SortedNatTree.values row6101 = Data.row6101 := rfl
theorem row6101_checked : SortedNatTree.contains row6101 = Data.row6101.contains := by
  rw [SortedNatTree.contains_eq_list row6101 row6101_ordered, row6101_values]
def row6113 : SortedNatTree.Tree := (.node 13679 (.node 8933 (.node 7001 (.node 6659 (.node 6353 (.node 6311 .empty .empty) .empty) (.node 6833 .empty .empty)) (.node 8663 (.node 7793 (.node 7691 .empty .empty) .empty) (.node 8849 .empty .empty))) (.node 11087 (.node 10949 (.node 9839 (.node 9413 .empty .empty) .empty) (.node 11027 .empty .empty)) (.node 12227 (.node 11471 .empty .empty) (.node 12959 .empty .empty)))) (.node 17519 (.node 15773 (.node 14621 (.node 14153 (.node 14051 .empty .empty) .empty) (.node 15077 .empty .empty)) (.node 16883 (.node 16661 (.node 15791 .empty .empty) .empty) (.node 17033 .empty .empty))) (.node 18899 (.node 17981 (.node 17957 (.node 17747 .empty .empty) .empty) (.node 18503 .empty .empty)) (.node 19157 (.node 19073 .empty .empty) (.node 19427 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6113_ordered : SortedNatTree.ordered row6113 = true := by decide
set_option maxRecDepth 4096 in
theorem row6113_values : SortedNatTree.values row6113 = Data.row6113 := rfl
theorem row6113_checked : SortedNatTree.contains row6113 = Data.row6113.contains := by
  rw [SortedNatTree.contains_eq_list row6113 row6113_ordered, row6113_values]
def row6121 : SortedNatTree.Tree := (.node 10273 (.node 8389 (.node 7489 (.node 6871 (.node 6619 .empty .empty) .empty) (.node 7993 .empty .empty)) (.node 9421 (.node 9181 (.node 9043 .empty .empty) .empty) (.node 9733 .empty .empty))) (.node 13627 (.node 11287 (.node 11113 (.node 10831 .empty .empty) .empty) (.node 11527 .empty .empty)) (.node 18973 (.node 18679 (.node 14401 .empty .empty) .empty) (.node 19429 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6121_ordered : SortedNatTree.ordered row6121 = true := by decide
set_option maxRecDepth 4096 in
theorem row6121_values : SortedNatTree.values row6121 = Data.row6121 := rfl
theorem row6121_checked : SortedNatTree.contains row6121 = Data.row6121.contains := by
  rw [SortedNatTree.contains_eq_list row6121 row6121_ordered, row6121_values]
def row6131 : SortedNatTree.Tree := (.node 13469 (.node 9281 (.node 7577 (.node 6947 (.node 6863 .empty .empty) (.node 7019 .empty .empty)) (.node 8849 (.node 8123 .empty .empty) (.node 9059 .empty .empty))) (.node 12323 (.node 10037 (.node 9689 .empty .empty) (.node 10937 .empty .empty)) (.node 13217 (.node 12491 .empty .empty) .empty))) (.node 17093 (.node 14699 (.node 14321 (.node 14057 .empty .empty) (.node 14561 .empty .empty)) (.node 16787 (.node 14771 .empty .empty) .empty)) (.node 18911 (.node 17627 (.node 17471 .empty .empty) (.node 18731 .empty .empty)) (.node 19541 (.node 19319 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6131_ordered : SortedNatTree.ordered row6131 = true := by decide
set_option maxRecDepth 4096 in
theorem row6131_values : SortedNatTree.values row6131 = Data.row6131 := rfl
theorem row6131_checked : SortedNatTree.contains row6131 = Data.row6131.contains := by
  rw [SortedNatTree.contains_eq_list row6131 row6131_ordered, row6131_values]
def row6133 : SortedNatTree.Tree := (.node 13411 (.node 9319 (.node 8353 (.node 7741 (.node 6871 .empty .empty) .empty) (.node 9001 (.node 8521 .empty .empty) .empty)) (.node 10639 (.node 10069 (.node 9661 .empty .empty) .empty) (.node 12781 (.node 11299 .empty .empty) .empty))) (.node 16657 (.node 15259 (.node 14071 (.node 13903 .empty .empty) .empty) (.node 16603 (.node 15973 .empty .empty) .empty)) (.node 18451 (.node 17389 (.node 16981 .empty .empty) .empty) (.node 19249 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6133_ordered : SortedNatTree.ordered row6133 = true := by decide
set_option maxRecDepth 4096 in
theorem row6133_values : SortedNatTree.values row6133 = Data.row6133 := rfl
theorem row6133_checked : SortedNatTree.contains row6133 = Data.row6133.contains := by
  rw [SortedNatTree.contains_eq_list row6133 row6133_ordered, row6133_values]
def row6143 : SortedNatTree.Tree := (.node 11783 (.node 8219 (.node 6833 (.node 6689 (.node 6311 .empty .empty) .empty) (.node 8171 (.node 7151 .empty .empty) .empty)) (.node 9437 (.node 8999 (.node 8963 .empty .empty) .empty) (.node 11681 (.node 10271 .empty .empty) .empty))) (.node 14639 (.node 12659 (.node 12203 (.node 11939 .empty .empty) .empty) (.node 13967 (.node 13499 .empty .empty) .empty)) (.node 16487 (.node 16433 (.node 15473 .empty .empty) .empty) (.node 17579 (.node 16763 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6143_ordered : SortedNatTree.ordered row6143 = true := by decide
set_option maxRecDepth 4096 in
theorem row6143_values : SortedNatTree.values row6143 = Data.row6143 := rfl
theorem row6143_checked : SortedNatTree.contains row6143 = Data.row6143.contains := by
  rw [SortedNatTree.contains_eq_list row6143 row6143_ordered, row6143_values]
def row6151 : SortedNatTree.Tree := (.node 12889 (.node 9463 (.node 8311 (.node 7933 (.node 7549 (.node 6571 .empty .empty) .empty) (.node 8221 .empty .empty)) (.node 8647 (.node 8329 .empty .empty) (.node 8893 .empty .empty))) (.node 10729 (.node 10267 (.node 9613 .empty .empty) (.node 10501 .empty .empty)) (.node 12451 (.node 11353 .empty .empty) (.node 12763 .empty .empty)))) (.node 16057 (.node 14293 (.node 13711 (.node 12907 .empty .empty) (.node 14029 .empty .empty)) (.node 14869 (.node 14827 .empty .empty) (.node 15493 .empty .empty))) (.node 17863 (.node 16633 (.node 16603 .empty .empty) (.node 16981 .empty .empty)) (.node 19333 (.node 18979 .empty .empty) (.node 19507 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6151_ordered : SortedNatTree.ordered row6151 = true := by decide
set_option maxRecDepth 4096 in
theorem row6151_values : SortedNatTree.values row6151 = Data.row6151 := rfl
theorem row6151_checked : SortedNatTree.contains row6151 = Data.row6151.contains := by
  rw [SortedNatTree.contains_eq_list row6151 row6151_ordered, row6151_values]
def row6163 : SortedNatTree.Tree := (.node 10531 (.node 9001 (.node 8167 (.node 6793 (.node 6451 .empty .empty) (.node 7867 .empty .empty)) (.node 8971 (.node 8713 .empty .empty) .empty)) (.node 9679 (.node 9349 (.node 9343 .empty .empty) .empty) (.node 10399 (.node 9733 .empty .empty) .empty))) (.node 14389 (.node 11287 (.node 11071 (.node 10663 .empty .empty) .empty) (.node 13183 (.node 12799 .empty .empty) .empty)) (.node 16447 (.node 15607 (.node 14437 .empty .empty) .empty) (.node 19609 (.node 17239 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6163_ordered : SortedNatTree.ordered row6163 = true := by decide
set_option maxRecDepth 4096 in
theorem row6163_values : SortedNatTree.values row6163 = Data.row6163 := rfl
theorem row6163_checked : SortedNatTree.contains row6163 = Data.row6163.contains := by
  rw [SortedNatTree.contains_eq_list row6163 row6163_ordered, row6163_values]
def row6173 : SortedNatTree.Tree := (.node 11969 (.node 9371 (.node 7853 (.node 7283 (.node 6581 .empty .empty) (.node 7691 .empty .empty)) (.node 8231 (.node 7901 .empty .empty) .empty)) (.node 10487 (.node 9719 (.node 9497 .empty .empty) .empty) (.node 10979 (.node 10601 .empty .empty) .empty))) (.node 16871 (.node 14783 (.node 13457 (.node 13007 .empty .empty) .empty) (.node 15887 (.node 15551 .empty .empty) .empty)) (.node 17573 (.node 17489 (.node 17471 .empty .empty) .empty) (.node 19403 (.node 18311 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6173_ordered : SortedNatTree.ordered row6173 = true := by decide
set_option maxRecDepth 4096 in
theorem row6173_values : SortedNatTree.values row6173 = Data.row6173 := rfl
theorem row6173_checked : SortedNatTree.contains row6173 = Data.row6173.contains := by
  rw [SortedNatTree.contains_eq_list row6173 row6173_ordered, row6173_values]
def row6197 : SortedNatTree.Tree := (.node 12377 (.node 7793 (.node 7151 (.node 6983 (.node 6389 .empty .empty) .empty) (.node 7547 .empty .empty)) (.node 10337 (.node 9209 .empty .empty) (.node 11831 .empty .empty))) (.node 15401 (.node 15359 (.node 14321 .empty .empty) (.node 15377 .empty .empty)) (.node 16829 (.node 16421 .empty .empty) (.node 19259 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6197_ordered : SortedNatTree.ordered row6197 = true := by decide
set_option maxRecDepth 4096 in
theorem row6197_values : SortedNatTree.values row6197 = Data.row6197 := rfl
theorem row6197_checked : SortedNatTree.contains row6197 = Data.row6197.contains := by
  rw [SortedNatTree.contains_eq_list row6197 row6197_ordered, row6197_values]
def row6199 : SortedNatTree.Tree := (.node 13477 (.node 11503 (.node 8863 (.node 7219 (.node 6673 .empty .empty) .empty) (.node 9787 .empty .empty)) (.node 12637 (.node 12253 .empty .empty) (.node 13003 .empty .empty))) (.node 16987 (.node 16189 (.node 14389 (.node 14143 .empty .empty) .empty) (.node 16903 .empty .empty)) (.node 18307 (.node 17239 .empty .empty) (.node 18919 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6199_ordered : SortedNatTree.ordered row6199 = true := by decide
set_option maxRecDepth 4096 in
theorem row6199_values : SortedNatTree.values row6199 = Data.row6199 := rfl
theorem row6199_checked : SortedNatTree.contains row6199 = Data.row6199.contains := by
  rw [SortedNatTree.contains_eq_list row6199 row6199_ordered, row6199_values]
def row6203 : SortedNatTree.Tree := (.node 11369 (.node 7823 (.node 7043 (.node 6551 (.node 6323 .empty .empty) (.node 6761 .empty .empty)) (.node 7643 (.node 7253 .empty .empty) .empty)) (.node 9473 (.node 9311 (.node 8693 .empty .empty) (.node 9377 .empty .empty)) (.node 10181 (.node 9629 .empty .empty) .empty))) (.node 18329 (.node 16097 (.node 12071 (.node 11621 .empty .empty) (.node 12743 .empty .empty)) (.node 17903 (.node 16811 .empty .empty) .empty)) (.node 19421 (.node 18749 (.node 18521 .empty .empty) (.node 19013 .empty .empty)) (.node 19727 (.node 19571 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6203_ordered : SortedNatTree.ordered row6203 = true := by decide
set_option maxRecDepth 4096 in
theorem row6203_values : SortedNatTree.values row6203 = Data.row6203 := rfl
theorem row6203_checked : SortedNatTree.contains row6203 = Data.row6203.contains := by
  rw [SortedNatTree.contains_eq_list row6203 row6203_ordered, row6203_values]
def row6211 : SortedNatTree.Tree := (.node 14173 (.node 9181 (.node 6883 (.node 6547 (.node 6277 .empty .empty) .empty) (.node 8887 (.node 7753 .empty .empty) .empty)) (.node 11503 (.node 10891 (.node 9613 .empty .empty) .empty) (.node 13099 .empty .empty))) (.node 18457 (.node 16759 (.node 15331 (.node 14551 .empty .empty) .empty) (.node 18229 .empty .empty)) (.node 19237 (.node 18787 (.node 18661 .empty .empty) .empty) (.node 19483 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6211_ordered : SortedNatTree.ordered row6211 = true := by decide
set_option maxRecDepth 4096 in
theorem row6211_values : SortedNatTree.values row6211 = Data.row6211 := rfl
theorem row6211_checked : SortedNatTree.contains row6211 = Data.row6211.contains := by
  rw [SortedNatTree.contains_eq_list row6211 row6211_ordered, row6211_values]
def row6217 : SortedNatTree.Tree := (.node 13633 (.node 8443 (.node 7933 (.node 6679 (.node 6481 .empty .empty) .empty) (.node 8329 (.node 8161 .empty .empty) .empty)) (.node 10357 (.node 9391 (.node 8863 .empty .empty) .empty) (.node 12919 (.node 11941 .empty .empty) .empty))) (.node 16411 (.node 15817 (.node 14563 (.node 14503 .empty .empty) .empty) (.node 16369 (.node 15901 .empty .empty) .empty)) (.node 18637 (.node 18127 (.node 17077 .empty .empty) .empty) (.node 19489 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6217_ordered : SortedNatTree.ordered row6217 = true := by decide
set_option maxRecDepth 4096 in
theorem row6217_values : SortedNatTree.values row6217 = Data.row6217 := rfl
theorem row6217_checked : SortedNatTree.contains row6217 = Data.row6217.contains := by
  rw [SortedNatTree.contains_eq_list row6217 row6217_ordered, row6217_values]
def row6221 : SortedNatTree.Tree := (.node 12923 (.node 8363 (.node 6971 (.node 6653 (.node 6599 .empty .empty) .empty) (.node 7577 .empty .empty)) (.node 9719 (.node 9221 (.node 9173 .empty .empty) .empty) (.node 10667 .empty .empty))) (.node 15107 (.node 14639 (.node 14489 (.node 14243 .empty .empty) .empty) (.node 14951 .empty .empty)) (.node 17471 (.node 16529 (.node 15401 .empty .empty) .empty) (.node 18593 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6221_ordered : SortedNatTree.ordered row6221 = true := by decide
set_option maxRecDepth 4096 in
theorem row6221_values : SortedNatTree.values row6221 = Data.row6221 := rfl
theorem row6221_checked : SortedNatTree.contains row6221 = Data.row6221.contains := by
  rw [SortedNatTree.contains_eq_list row6221 row6221_ordered, row6221_values]
def row6229 : SortedNatTree.Tree := (.node 12211 (.node 9283 (.node 8233 (.node 7297 (.node 6277 .empty .empty) (.node 8167 .empty .empty)) (.node 8893 (.node 8839 .empty .empty) .empty)) (.node 9967 (.node 9547 (.node 9421 .empty .empty) .empty) (.node 11953 (.node 10333 .empty .empty) .empty))) (.node 15541 (.node 14437 (.node 13171 (.node 12253 .empty .empty) (.node 13789 .empty .empty)) (.node 15289 (.node 14797 .empty .empty) .empty)) (.node 17599 (.node 16633 (.node 15901 .empty .empty) .empty) (.node 19081 (.node 19069 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6229_ordered : SortedNatTree.ordered row6229 = true := by decide
set_option maxRecDepth 4096 in
theorem row6229_values : SortedNatTree.values row6229 = Data.row6229 := rfl
theorem row6229_checked : SortedNatTree.contains row6229 = Data.row6229.contains := by
  rw [SortedNatTree.contains_eq_list row6229 row6229_ordered, row6229_values]
def row6247 : SortedNatTree.Tree := (.node 12511 (.node 8599 (.node 7549 (.node 6709 (.node 6691 .empty .empty) .empty) (.node 8161 (.node 7879 .empty .empty) .empty)) (.node 9187 (.node 9007 (.node 8641 .empty .empty) .empty) (.node 12421 (.node 10651 .empty .empty) .empty))) (.node 17257 (.node 15787 (.node 14437 (.node 13183 .empty .empty) .empty) (.node 17167 (.node 15913 .empty .empty) .empty)) (.node 18181 (.node 17761 (.node 17299 .empty .empty) .empty) (.node 19759 (.node 18379 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6247_ordered : SortedNatTree.ordered row6247 = true := by decide
set_option maxRecDepth 4096 in
theorem row6247_values : SortedNatTree.values row6247 = Data.row6247 := rfl
theorem row6247_checked : SortedNatTree.contains row6247 = Data.row6247.contains := by
  rw [SortedNatTree.contains_eq_list row6247 row6247_ordered, row6247_values]
def row6257 : SortedNatTree.Tree := (.node 9833 (.node 8693 (.node 8543 (.node 8387 (.node 7607 .empty .empty) .empty) (.node 8627 .empty .empty)) (.node 9719 (.node 9551 .empty .empty) (.node 9791 .empty .empty))) (.node 13709 (.node 11777 (.node 11549 (.node 10979 .empty .empty) .empty) (.node 13217 .empty .empty)) (.node 15383 (.node 14519 .empty .empty) (.node 15683 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6257_ordered : SortedNatTree.ordered row6257 = true := by decide
set_option maxRecDepth 4096 in
theorem row6257_values : SortedNatTree.values row6257 = Data.row6257 := rfl
theorem row6257_checked : SortedNatTree.contains row6257 = Data.row6257.contains := by
  rw [SortedNatTree.contains_eq_list row6257 row6257_ordered, row6257_values]
def row6263 : SortedNatTree.Tree := (.node 11177 (.node 8819 (.node 7499 (.node 6389 (.node 6359 .empty .empty) (.node 7121 .empty .empty)) (.node 8297 (.node 7919 .empty .empty) .empty)) (.node 10247 (.node 9791 (.node 9539 .empty .empty) .empty) (.node 10859 (.node 10337 .empty .empty) .empty))) (.node 14369 (.node 12497 (.node 11489 (.node 11261 .empty .empty) .empty) (.node 14321 (.node 13799 .empty .empty) .empty)) (.node 15401 (.node 15329 (.node 14831 .empty .empty) .empty) (.node 17183 (.node 16619 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6263_ordered : SortedNatTree.ordered row6263 = true := by decide
set_option maxRecDepth 4096 in
theorem row6263_values : SortedNatTree.values row6263 = Data.row6263 := rfl
theorem row6263_checked : SortedNatTree.contains row6263 = Data.row6263.contains := by
  rw [SortedNatTree.contains_eq_list row6263 row6263_ordered, row6263_values]
def row6269 : SortedNatTree.Tree := (.node 10883 (.node 8291 (.node 6899 (.node 6779 (.node 6701 .empty .empty) .empty) (.node 7883 (.node 7457 .empty .empty) .empty)) (.node 9137 (.node 8849 (.node 8369 .empty .empty) .empty) (.node 10169 (.node 9431 .empty .empty) .empty))) (.node 14303 (.node 12161 (.node 11939 (.node 11489 .empty .empty) .empty) (.node 13997 (.node 12653 .empty .empty) .empty)) (.node 17519 (.node 16349 (.node 15803 .empty .empty) .empty) (.node 19289 (.node 17747 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6269_ordered : SortedNatTree.ordered row6269 = true := by decide
set_option maxRecDepth 4096 in
theorem row6269_values : SortedNatTree.values row6269 = Data.row6269 := rfl
theorem row6269_checked : SortedNatTree.contains row6269 = Data.row6269.contains := by
  rw [SortedNatTree.contains_eq_list row6269 row6269_ordered, row6269_values]
def row6271 : SortedNatTree.Tree := (.node 14143 (.node 11047 (.node 8779 (.node 8011 (.node 6373 .empty .empty) (.node 8641 .empty .empty)) (.node 9661 (.node 9187 .empty .empty) (.node 10711 .empty .empty))) (.node 12379 (.node 11257 (.node 11119 .empty .empty) (.node 11743 .empty .empty)) (.node 13693 (.node 12433 .empty .empty) (.node 13723 .empty .empty)))) (.node 16993 (.node 15679 (.node 14503 (.node 14347 .empty .empty) (.node 15277 .empty .empty)) (.node 16069 (.node 15823 .empty .empty) (.node 16573 .empty .empty))) (.node 18397 (.node 17581 (.node 17209 .empty .empty) (.node 18043 .empty .empty)) (.node 19309 (.node 18637 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6271_ordered : SortedNatTree.ordered row6271 = true := by decide
set_option maxRecDepth 4096 in
theorem row6271_values : SortedNatTree.values row6271 = Data.row6271 := rfl
theorem row6271_checked : SortedNatTree.contains row6271 = Data.row6271.contains := by
  rw [SortedNatTree.contains_eq_list row6271 row6271_ordered, row6271_values]
def row6277 : SortedNatTree.Tree := (.node 13267 (.node 9091 (.node 8269 (.node 7681 (.node 7321 .empty .empty) .empty) (.node 8623 (.node 8563 .empty .empty) .empty)) (.node 11779 (.node 11047 (.node 9109 .empty .empty) .empty) (.node 12451 (.node 12391 .empty .empty) .empty))) (.node 15541 (.node 14407 (.node 14173 (.node 13477 .empty .empty) .empty) (.node 15511 (.node 15493 .empty .empty) .empty)) (.node 18457 (.node 18199 (.node 17317 .empty .empty) .empty) (.node 19699 (.node 19501 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6277_ordered : SortedNatTree.ordered row6277 = true := by decide
set_option maxRecDepth 4096 in
theorem row6277_values : SortedNatTree.values row6277 = Data.row6277 := rfl
theorem row6277_checked : SortedNatTree.contains row6277 = Data.row6277.contains := by
  rw [SortedNatTree.contains_eq_list row6277 row6277_ordered, row6277_values]
def row6287 : SortedNatTree.Tree := (.node 13037 (.node 8609 (.node 8147 (.node 7673 (.node 6911 .empty .empty) (.node 8009 .empty .empty)) (.node 8273 (.node 8219 .empty .empty) .empty)) (.node 10079 (.node 9461 (.node 9137 .empty .empty) .empty) (.node 11813 (.node 11579 .empty .empty) .empty))) (.node 15467 (.node 14771 (.node 14057 (.node 13457 .empty .empty) (.node 14759 .empty .empty)) (.node 15401 (.node 15299 .empty .empty) .empty)) (.node 16481 (.node 16103 (.node 15887 .empty .empty) .empty) (.node 18131 (.node 17909 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6287_ordered : SortedNatTree.ordered row6287 = true := by decide
set_option maxRecDepth 4096 in
theorem row6287_values : SortedNatTree.values row6287 = Data.row6287 := rfl
theorem row6287_checked : SortedNatTree.contains row6287 = Data.row6287.contains := by
  rw [SortedNatTree.contains_eq_list row6287 row6287_ordered, row6287_values]
def row6299 : SortedNatTree.Tree := (.node 12641 (.node 10067 (.node 8681 (.node 8147 (.node 7589 (.node 7283 .empty .empty) .empty) (.node 8501 .empty .empty)) (.node 9887 (.node 9857 .empty .empty) (.node 10007 .empty .empty))) (.node 11483 (.node 10601 (.node 10133 (.node 10079 .empty .empty) .empty) (.node 10937 .empty .empty)) (.node 12011 (.node 11801 .empty .empty) (.node 12347 .empty .empty)))) (.node 16349 (.node 13763 (.node 13499 (.node 13469 (.node 13451 .empty .empty) .empty) (.node 13679 .empty .empty)) (.node 15131 (.node 13931 .empty .empty) (.node 15581 .empty .empty))) (.node 18959 (.node 16979 (.node 16871 (.node 16649 .empty .empty) .empty) (.node 17027 .empty .empty)) (.node 19457 (.node 19421 .empty .empty) (.node 19583 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6299_ordered : SortedNatTree.ordered row6299 = true := by decide
set_option maxRecDepth 4096 in
theorem row6299_values : SortedNatTree.values row6299 = Data.row6299 := rfl
theorem row6299_checked : SortedNatTree.contains row6299 = Data.row6299.contains := by
  rw [SortedNatTree.contains_eq_list row6299 row6299_ordered, row6299_values]
def row6301 : SortedNatTree.Tree := (.node 11491 (.node 8233 (.node 7321 (.node 7027 (.node 6607 .empty .empty) (.node 7039 .empty .empty)) (.node 7639 (.node 7417 .empty .empty) (.node 8101 .empty .empty))) (.node 9883 (.node 9127 (.node 8623 .empty .empty) (.node 9739 .empty .empty)) (.node 10069 (.node 9949 .empty .empty) (.node 10513 .empty .empty)))) (.node 15991 (.node 13513 (.node 12211 (.node 11731 .empty .empty) (.node 13339 .empty .empty)) (.node 15091 (.node 14293 .empty .empty) (.node 15583 .empty .empty))) (.node 17011 (.node 16369 (.node 16141 .empty .empty) (.node 16603 .empty .empty)) (.node 18367 (.node 17077 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6301_ordered : SortedNatTree.ordered row6301 = true := by decide
set_option maxRecDepth 4096 in
theorem row6301_values : SortedNatTree.values row6301 = Data.row6301 := rfl
theorem row6301_checked : SortedNatTree.contains row6301 = Data.row6301.contains := by
  rw [SortedNatTree.contains_eq_list row6301 row6301_ordered, row6301_values]
def row6311 : SortedNatTree.Tree := (.node 13001 (.node 8543 (.node 6863 (.node 6737 (.node 6491 .empty .empty) .empty) (.node 8423 (.node 6947 .empty .empty) .empty)) (.node 10427 (.node 10169 (.node 9941 .empty .empty) .empty) (.node 11909 .empty .empty))) (.node 17021 (.node 16427 (.node 15131 (.node 14009 .empty .empty) .empty) (.node 16487 .empty .empty)) (.node 19001 (.node 18671 (.node 17609 .empty .empty) .empty) (.node 19661 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6311_ordered : SortedNatTree.ordered row6311 = true := by decide
set_option maxRecDepth 4096 in
theorem row6311_values : SortedNatTree.values row6311 = Data.row6311 := rfl
theorem row6311_checked : SortedNatTree.contains row6311 = Data.row6311.contains := by
  rw [SortedNatTree.contains_eq_list row6311 row6311_ordered, row6311_values]
def row6317 : SortedNatTree.Tree := (.node 14009 (.node 8543 (.node 7607 (.node 6983 (.node 6599 .empty .empty) (.node 7529 .empty .empty)) (.node 8039 (.node 7853 .empty .empty) .empty)) (.node 10133 (.node 9491 (.node 9323 .empty .empty) .empty) (.node 11939 (.node 11777 .empty .empty) .empty))) (.node 17789 (.node 15809 (.node 14549 (.node 14423 .empty .empty) (.node 15101 .empty .empty)) (.node 16301 (.node 16223 .empty .empty) .empty)) (.node 18413 (.node 18041 (.node 17987 .empty .empty) .empty) (.node 19259 (.node 19211 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6317_ordered : SortedNatTree.ordered row6317 = true := by decide
set_option maxRecDepth 4096 in
theorem row6317_values : SortedNatTree.values row6317 = Data.row6317 := rfl
theorem row6317_checked : SortedNatTree.contains row6317 = Data.row6317.contains := by
  rw [SortedNatTree.contains_eq_list row6317 row6317_ordered, row6317_values]
def row6323 : SortedNatTree.Tree := (.node 10169 (.node 8741 (.node 7523 (.node 7019 (.node 6491 .empty .empty) .empty) (.node 8237 (.node 7919 .empty .empty) .empty)) (.node 9413 (.node 9311 (.node 8849 .empty .empty) .empty) (.node 9833 .empty .empty))) (.node 13109 (.node 12347 (.node 11003 (.node 10271 .empty .empty) .empty) (.node 12917 (.node 12809 .empty .empty) .empty)) (.node 17333 (.node 16823 (.node 13337 .empty .empty) .empty) (.node 18041 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6323_ordered : SortedNatTree.ordered row6323 = true := by decide
set_option maxRecDepth 4096 in
theorem row6323_values : SortedNatTree.values row6323 = Data.row6323 := rfl
theorem row6323_checked : SortedNatTree.contains row6323 = Data.row6323.contains := by
  rw [SortedNatTree.contains_eq_list row6323 row6323_ordered, row6323_values]
def row6329 : SortedNatTree.Tree := (.node 11927 (.node 7937 (.node 7331 (.node 6983 (.node 6551 .empty .empty) .empty) (.node 7907 (.node 7589 .empty .empty) .empty)) (.node 10331 (.node 9851 (.node 9377 .empty .empty) .empty) (.node 11789 (.node 10739 .empty .empty) .empty))) (.node 16607 (.node 14561 (.node 13037 (.node 12281 .empty .empty) .empty) (.node 16421 (.node 14627 .empty .empty) .empty)) (.node 19121 (.node 18287 (.node 16943 .empty .empty) .empty) (.node 19457 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6329_ordered : SortedNatTree.ordered row6329 = true := by decide
set_option maxRecDepth 4096 in
theorem row6329_values : SortedNatTree.values row6329 = Data.row6329 := rfl
theorem row6329_checked : SortedNatTree.contains row6329 = Data.row6329.contains := by
  rw [SortedNatTree.contains_eq_list row6329 row6329_ordered, row6329_values]
def row6337 : SortedNatTree.Tree := (.node 12577 (.node 9907 (.node 8443 (.node 7933 (.node 6991 (.node 6949 .empty .empty) .empty) (.node 8101 .empty .empty)) (.node 9661 (.node 9013 .empty .empty) (.node 9769 .empty .empty))) (.node 11149 (.node 10243 (.node 10159 (.node 10039 .empty .empty) .empty) (.node 10459 .empty .empty)) (.node 11863 (.node 11779 .empty .empty) (.node 12511 .empty .empty)))) (.node 16879 (.node 14197 (.node 13291 (.node 12823 (.node 12613 .empty .empty) .empty) (.node 14173 .empty .empty)) (.node 15889 (.node 15583 .empty .empty) (.node 16249 .empty .empty))) (.node 17911 (.node 17191 (.node 17107 .empty .empty) (.node 17683 .empty .empty)) (.node 19237 (.node 17989 .empty .empty) (.node 19441 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6337_ordered : SortedNatTree.ordered row6337 = true := by decide
set_option maxRecDepth 4096 in
theorem row6337_values : SortedNatTree.values row6337 = Data.row6337 := rfl
theorem row6337_checked : SortedNatTree.contains row6337 = Data.row6337.contains := by
  rw [SortedNatTree.contains_eq_list row6337 row6337_ordered, row6337_values]
def row6343 : SortedNatTree.Tree := (.node 14389 (.node 9949 (.node 7867 (.node 7669 (.node 7039 .empty .empty) .empty) (.node 9697 (.node 9661 .empty .empty) .empty)) (.node 11437 (.node 11299 (.node 10321 .empty .empty) .empty) (.node 14143 .empty .empty))) (.node 16381 (.node 15259 (.node 14851 (.node 14827 .empty .empty) .empty) (.node 15277 .empty .empty)) (.node 17839 (.node 17581 (.node 17539 .empty .empty) .empty) (.node 19429 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6343_ordered : SortedNatTree.ordered row6343 = true := by decide
set_option maxRecDepth 4096 in
theorem row6343_values : SortedNatTree.values row6343 = Data.row6343 := rfl
theorem row6343_checked : SortedNatTree.contains row6343 = Data.row6343.contains := by
  rw [SortedNatTree.contains_eq_list row6343 row6343_ordered, row6343_values]
def row6353 : SortedNatTree.Tree := (.node 12911 (.node 11321 (.node 9467 (.node 7949 (.node 6359 .empty .empty) .empty) (.node 9923 (.node 9887 .empty .empty) .empty)) (.node 12323 (.node 12239 (.node 11597 .empty .empty) .empty) (.node 12497 .empty .empty))) (.node 16349 (.node 15017 (.node 14249 (.node 13037 .empty .empty) .empty) (.node 15299 .empty .empty)) (.node 18797 (.node 18701 (.node 17327 .empty .empty) .empty) (.node 19031 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6353_ordered : SortedNatTree.ordered row6353 = true := by decide
set_option maxRecDepth 4096 in
theorem row6353_values : SortedNatTree.values row6353 = Data.row6353 := rfl
theorem row6353_checked : SortedNatTree.contains row6353 = Data.row6353.contains := by
  rw [SortedNatTree.contains_eq_list row6353 row6353_ordered, row6353_values]
def row6359 : SortedNatTree.Tree := (.node 11657 (.node 7577 (.node 6761 (.node 6737 .empty .empty) (.node 7019 .empty .empty)) (.node 10139 (.node 8123 .empty .empty) (.node 10973 .empty .empty))) (.node 16649 (.node 15671 (.node 15299 .empty .empty) (.node 15767 .empty .empty)) (.node 18587 (.node 17231 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row6359_ordered : SortedNatTree.ordered row6359 = true := by decide
set_option maxRecDepth 4096 in
theorem row6359_values : SortedNatTree.values row6359 = Data.row6359 := rfl
theorem row6359_checked : SortedNatTree.contains row6359 = Data.row6359.contains := by
  rw [SortedNatTree.contains_eq_list row6359 row6359_ordered, row6359_values]
def row6361 : SortedNatTree.Tree := (.node 10627 (.node 8647 (.node 7507 (.node 6673 (.node 6481 .empty .empty) (.node 7069 .empty .empty)) (.node 8521 (.node 8389 .empty .empty) .empty)) (.node 9547 (.node 8941 (.node 8923 .empty .empty) (.node 9109 .empty .empty)) (.node 9973 (.node 9931 .empty .empty) .empty))) (.node 13249 (.node 11251 (.node 10957 (.node 10789 .empty .empty) (.node 11059 .empty .empty)) (.node 12973 (.node 12739 .empty .empty) .empty)) (.node 17239 (.node 15373 (.node 13669 .empty .empty) .empty) (.node 18859 (.node 18523 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6361_ordered : SortedNatTree.ordered row6361 = true := by decide
set_option maxRecDepth 4096 in
theorem row6361_values : SortedNatTree.values row6361 = Data.row6361 := rfl
theorem row6361_checked : SortedNatTree.contains row6361 = Data.row6361.contains := by
  rw [SortedNatTree.contains_eq_list row6361 row6361_ordered, row6361_values]
def row6367 : SortedNatTree.Tree := (.node 12919 (.node 10069 (.node 8353 (.node 6793 (.node 6703 .empty .empty) (.node 7477 .empty .empty)) (.node 9901 (.node 9619 .empty .empty) .empty)) (.node 11587 (.node 10369 (.node 10099 .empty .empty) (.node 10861 .empty .empty)) (.node 12829 (.node 12511 .empty .empty) .empty))) (.node 15907 (.node 14479 (.node 14341 (.node 13267 .empty .empty) (.node 14347 .empty .empty)) (.node 15661 (.node 15307 .empty .empty) .empty)) (.node 18133 (.node 16699 (.node 16111 .empty .empty) (.node 17377 .empty .empty)) (.node 18553 (.node 18169 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6367_ordered : SortedNatTree.ordered row6367 = true := by decide
set_option maxRecDepth 4096 in
theorem row6367_values : SortedNatTree.values row6367 = Data.row6367 := rfl
theorem row6367_checked : SortedNatTree.contains row6367 = Data.row6367.contains := by
  rw [SortedNatTree.contains_eq_list row6367 row6367_ordered, row6367_values]
def row6373 : SortedNatTree.Tree := (.node 11251 (.node 7699 (.node 7459 (.node 6781 (.node 6661 .empty .empty) .empty) (.node 7507 .empty .empty)) (.node 9967 (.node 8581 (.node 8167 .empty .empty) .empty) (.node 11083 .empty .empty))) (.node 15733 (.node 14737 (.node 13291 (.node 11497 .empty .empty) .empty) (.node 15361 .empty .empty)) (.node 16993 (.node 16333 (.node 16063 .empty .empty) .empty) (.node 19429 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6373_ordered : SortedNatTree.ordered row6373 = true := by decide
set_option maxRecDepth 4096 in
theorem row6373_values : SortedNatTree.values row6373 = Data.row6373 := rfl
theorem row6373_checked : SortedNatTree.contains row6373 = Data.row6373.contains := by
  rw [SortedNatTree.contains_eq_list row6373 row6373_ordered, row6373_values]
def row6379 : SortedNatTree.Tree := (.node 12553 (.node 9787 (.node 8233 (.node 7057 (.node 6703 .empty .empty) (.node 8017 .empty .empty)) (.node 9679 (.node 9133 .empty .empty) .empty)) (.node 11059 (.node 10639 (.node 9811 .empty .empty) .empty) (.node 11503 (.node 11467 .empty .empty) .empty))) (.node 16561 (.node 14281 (.node 12721 (.node 12619 .empty .empty) (.node 12907 .empty .empty)) (.node 15121 (.node 14779 .empty .empty) .empty)) (.node 19069 (.node 17977 (.node 17551 .empty .empty) .empty) (.node 19423 (.node 19231 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6379_ordered : SortedNatTree.ordered row6379 = true := by decide
set_option maxRecDepth 4096 in
theorem row6379_values : SortedNatTree.values row6379 = Data.row6379 := rfl
theorem row6379_checked : SortedNatTree.contains row6379 = Data.row6379.contains := by
  rw [SortedNatTree.contains_eq_list row6379 row6379_ordered, row6379_values]
def row6389 : SortedNatTree.Tree := (.node 11699 (.node 8111 (.node 7523 (.node 7481 (.node 6473 .empty .empty) .empty) (.node 7937 .empty .empty)) (.node 10037 (.node 8423 (.node 8363 .empty .empty) .empty) (.node 11027 .empty .empty))) (.node 14321 (.node 13859 (.node 13679 (.node 12941 .empty .empty) .empty) (.node 13883 .empty .empty)) (.node 16829 (.node 16361 .empty .empty) (.node 18131 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6389_ordered : SortedNatTree.ordered row6389 = true := by decide
set_option maxRecDepth 4096 in
theorem row6389_values : SortedNatTree.values row6389 = Data.row6389 := rfl
theorem row6389_checked : SortedNatTree.contains row6389 = Data.row6389.contains := by
  rw [SortedNatTree.contains_eq_list row6389 row6389_ordered, row6389_values]
def row6397 : SortedNatTree.Tree := (.node 13003 (.node 9283 (.node 8179 (.node 7213 (.node 6793 .empty .empty) (.node 7321 .empty .empty)) (.node 8443 (.node 8419 .empty .empty) (.node 8821 .empty .empty))) (.node 11497 (.node 10399 (.node 9661 .empty .empty) (.node 10993 .empty .empty)) (.node 12979 (.node 12739 .empty .empty) .empty))) (.node 16141 (.node 14797 (.node 14389 (.node 13903 .empty .empty) (.node 14461 .empty .empty)) (.node 15511 (.node 15319 .empty .empty) .empty)) (.node 18553 (.node 17569 (.node 16381 .empty .empty) (.node 18253 .empty .empty)) (.node 19501 (.node 19237 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6397_ordered : SortedNatTree.ordered row6397 = true := by decide
set_option maxRecDepth 4096 in
theorem row6397_values : SortedNatTree.values row6397 = Data.row6397 := rfl
theorem row6397_checked : SortedNatTree.contains row6397 = Data.row6397.contains := by
  rw [SortedNatTree.contains_eq_list row6397 row6397_ordered, row6397_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
