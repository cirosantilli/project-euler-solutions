import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row3691 : SortedNatTree.Tree := (.node 11317 (.node 7789 (.node 6337 (.node 4177 (.node 4153 (.node 4051 .empty .empty) .empty) (.node 5419 (.node 4561 .empty .empty) .empty)) (.node 7561 (.node 7477 (.node 7333 .empty .empty) .empty) (.node 7669 .empty .empty))) (.node 8863 (.node 8521 (.node 8419 (.node 8377 .empty .empty) .empty) (.node 8731 .empty .empty)) (.node 9433 (.node 9319 (.node 9013 .empty .empty) .empty) (.node 10369 .empty .empty)))) (.node 15919 (.node 13291 (.node 12343 (.node 12157 (.node 11923 .empty .empty) .empty) (.node 13267 (.node 12739 .empty .empty) .empty)) (.node 15601 (.node 15511 (.node 14071 .empty .empty) .empty) (.node 15787 .empty .empty))) (.node 21001 (.node 17971 (.node 17239 (.node 16603 .empty .empty) .empty) (.node 19381 .empty .empty)) (.node 21673 (.node 21481 (.node 21397 .empty .empty) .empty) (.node 22111 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3691_ordered : SortedNatTree.ordered row3691 = true := by decide
set_option maxRecDepth 4096 in
theorem row3691_values : SortedNatTree.values row3691 = Data.row3691 := rfl
theorem row3691_checked : SortedNatTree.contains row3691 = Data.row3691.contains := by
  rw [SortedNatTree.contains_eq_list row3691 row3691_ordered, row3691_values]
def row3697 : SortedNatTree.Tree := (.node 9619 (.node 7459 (.node 5749 (.node 5209 (.node 4603 (.node 3919 .empty .empty) .empty) (.node 5431 .empty .empty)) (.node 6469 (.node 6007 .empty .empty) (.node 6673 .empty .empty))) (.node 8803 (.node 8563 (.node 7879 (.node 7867 .empty .empty) .empty) (.node 8647 .empty .empty)) (.node 9043 (.node 8941 .empty .empty) (.node 9181 .empty .empty)))) (.node 16843 (.node 11689 (.node 11059 (.node 10651 (.node 10333 .empty .empty) .empty) (.node 11311 .empty .empty)) (.node 14479 (.node 13873 .empty .empty) (.node 15739 .empty .empty))) (.node 19069 (.node 18211 (.node 17443 .empty .empty) (.node 18379 .empty .empty)) (.node 20161 (.node 19219 .empty .empty) (.node 21277 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3697_ordered : SortedNatTree.ordered row3697 = true := by decide
set_option maxRecDepth 4096 in
theorem row3697_values : SortedNatTree.values row3697 = Data.row3697 := rfl
theorem row3697_checked : SortedNatTree.contains row3697 = Data.row3697.contains := by
  rw [SortedNatTree.contains_eq_list row3697 row3697_ordered, row3697_values]
def row3701 : SortedNatTree.Tree := (.node 9623 (.node 5351 (.node 4919 (.node 4493 (.node 4283 (.node 3797 .empty .empty) .empty) (.node 4877 .empty .empty)) (.node 5081 (.node 4973 (.node 4931 .empty .empty) .empty) (.node 5309 .empty .empty))) (.node 7829 (.node 6869 (.node 6299 (.node 5501 .empty .empty) .empty) (.node 6911 .empty .empty)) (.node 9161 (.node 8609 (.node 8543 .empty .empty) .empty) (.node 9473 .empty .empty)))) (.node 14951 (.node 13463 (.node 11867 (.node 10781 (.node 9791 .empty .empty) .empty) (.node 12539 .empty .empty)) (.node 14369 (.node 13859 (.node 13697 .empty .empty) .empty) (.node 14639 .empty .empty))) (.node 19073 (.node 16301 (.node 16127 (.node 16001 .empty .empty) .empty) (.node 17333 .empty .empty)) (.node 21821 (.node 21491 .empty .empty) (.node 22073 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3701_ordered : SortedNatTree.ordered row3701 = true := by decide
set_option maxRecDepth 4096 in
theorem row3701_values : SortedNatTree.values row3701 = Data.row3701 := rfl
theorem row3701_checked : SortedNatTree.contains row3701 = Data.row3701.contains := by
  rw [SortedNatTree.contains_eq_list row3701 row3701_ordered, row3701_values]
def row3709 : SortedNatTree.Tree := (.node 13339 (.node 8389 (.node 6991 (.node 4261 (.node 4231 .empty .empty) (.node 4621 .empty .empty)) (.node 8377 (.node 7369 .empty .empty) .empty)) (.node 11257 (.node 10597 (.node 8779 .empty .empty) (.node 10711 .empty .empty)) (.node 12601 (.node 11743 .empty .empty) .empty))) (.node 18481 (.node 15817 (.node 14503 (.node 13831 .empty .empty) (.node 15679 .empty .empty)) (.node 18211 (.node 17377 .empty .empty) .empty)) (.node 20233 (.node 19687 (.node 19603 .empty .empty) (.node 20047 .empty .empty)) (.node 21319 (.node 20593 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3709_ordered : SortedNatTree.ordered row3709 = true := by decide
set_option maxRecDepth 4096 in
theorem row3709_values : SortedNatTree.values row3709 = Data.row3709 := rfl
theorem row3709_checked : SortedNatTree.contains row3709 = Data.row3709.contains := by
  rw [SortedNatTree.contains_eq_list row3709 row3709_ordered, row3709_values]
def row3719 : SortedNatTree.Tree := (.node 13781 (.node 10103 (.node 5591 (.node 5273 (.node 4271 .empty .empty) (.node 5297 .empty .empty)) (.node 8663 (.node 6791 .empty .empty) .empty)) (.node 13187 (.node 12899 (.node 11621 .empty .empty) .empty) (.node 13697 (.node 13259 .empty .empty) .empty))) (.node 17207 (.node 16001 (.node 15017 (.node 14489 .empty .empty) .empty) (.node 16487 (.node 16067 .empty .empty) .empty)) (.node 19157 (.node 17939 (.node 17417 .empty .empty) .empty) (.node 22157 (.node 19739 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3719_ordered : SortedNatTree.ordered row3719 = true := by decide
set_option maxRecDepth 4096 in
theorem row3719_values : SortedNatTree.values row3719 = Data.row3719 := rfl
theorem row3719_checked : SortedNatTree.contains row3719 = Data.row3719.contains := by
  rw [SortedNatTree.contains_eq_list row3719 row3719_ordered, row3719_values]
def row3727 : SortedNatTree.Tree := (.node 10993 (.node 7699 (.node 5659 (.node 4639 (.node 4129 .empty .empty) (.node 5011 .empty .empty)) (.node 6949 (.node 6037 .empty .empty) (.node 6961 .empty .empty))) (.node 8887 (.node 8623 (.node 8419 .empty .empty) (.node 8713 .empty .empty)) (.node 9817 (.node 9049 .empty .empty) .empty))) (.node 16189 (.node 13381 (.node 12073 (.node 11743 .empty .empty) (.node 13171 .empty .empty)) (.node 13963 (.node 13921 .empty .empty) .empty)) (.node 19429 (.node 16987 (.node 16699 .empty .empty) (.node 17959 .empty .empty)) (.node 20899 (.node 19507 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3727_ordered : SortedNatTree.ordered row3727 = true := by decide
set_option maxRecDepth 4096 in
theorem row3727_values : SortedNatTree.values row3727 = Data.row3727 := rfl
theorem row3727_checked : SortedNatTree.contains row3727 = Data.row3727.contains := by
  rw [SortedNatTree.contains_eq_list row3727 row3727_ordered, row3727_values]
def row3733 : SortedNatTree.Tree := (.node 11971 (.node 8461 (.node 6451 (.node 6151 (.node 5521 (.node 4657 .empty .empty) .empty) (.node 6421 .empty .empty)) (.node 7687 (.node 6529 .empty .empty) (.node 8089 .empty .empty))) (.node 10627 (.node 10159 (.node 9349 .empty .empty) (.node 10513 .empty .empty)) (.node 11779 (.node 10753 .empty .empty) (.node 11941 .empty .empty)))) (.node 15889 (.node 14629 (.node 13309 (.node 12697 (.node 12097 .empty .empty) .empty) (.node 13399 .empty .empty)) (.node 15541 (.node 15091 .empty .empty) (.node 15643 .empty .empty))) (.node 17257 (.node 16231 (.node 15973 .empty .empty) (.node 16981 .empty .empty)) (.node 21121 (.node 20641 .empty .empty) (.node 22273 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3733_ordered : SortedNatTree.ordered row3733 = true := by decide
set_option maxRecDepth 4096 in
theorem row3733_values : SortedNatTree.values row3733 = Data.row3733 := rfl
theorem row3733_checked : SortedNatTree.contains row3733 = Data.row3733.contains := by
  rw [SortedNatTree.contains_eq_list row3733 row3733_ordered, row3733_values]
def row3739 : SortedNatTree.Tree := (.node 14011 (.node 8233 (.node 6637 (.node 5233 (.node 5107 (.node 4099 .empty .empty) .empty) (.node 5851 .empty .empty)) (.node 6763 (.node 6679 .empty .empty) (.node 6871 .empty .empty))) (.node 9829 (.node 9151 (.node 8539 (.node 8353 .empty .empty) .empty) (.node 9781 .empty .empty)) (.node 12379 (.node 10993 .empty .empty) (.node 13807 .empty .empty)))) (.node 19333 (.node 17377 (.node 16363 (.node 14869 (.node 14071 .empty .empty) .empty) (.node 16729 .empty .empty)) (.node 17749 (.node 17683 .empty .empty) (.node 19237 .empty .empty))) (.node 20749 (.node 20563 (.node 20023 .empty .empty) (.node 20731 .empty .empty)) (.node 21283 (.node 21139 .empty .empty) (.node 22291 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3739_ordered : SortedNatTree.ordered row3739 = true := by decide
set_option maxRecDepth 4096 in
theorem row3739_values : SortedNatTree.values row3739 = Data.row3739 := rfl
theorem row3739_checked : SortedNatTree.contains row3739 = Data.row3739.contains := by
  rw [SortedNatTree.contains_eq_list row3739 row3739_ordered, row3739_values]
def row3761 : SortedNatTree.Tree := (.node 10037 (.node 7499 (.node 4799 (.node 4229 (.node 3767 .empty .empty) (.node 4391 .empty .empty)) (.node 5843 (.node 5807 .empty .empty) (.node 6287 .empty .empty))) (.node 8573 (.node 7793 (.node 7523 .empty .empty) (.node 8369 .empty .empty)) (.node 9209 (.node 8783 .empty .empty) (.node 9749 .empty .empty)))) (.node 16553 (.node 12413 (.node 11447 (.node 11171 .empty .empty) (.node 12263 .empty .empty)) (.node 13841 (.node 13691 .empty .empty) (.node 15359 .empty .empty))) (.node 19157 (.node 18059 (.node 17921 .empty .empty) (.node 18233 .empty .empty)) (.node 19661 (.node 19259 .empty .empty) (.node 20123 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3761_ordered : SortedNatTree.ordered row3761 = true := by decide
set_option maxRecDepth 4096 in
theorem row3761_values : SortedNatTree.values row3761 = Data.row3761 := rfl
theorem row3761_checked : SortedNatTree.contains row3761 = Data.row3761.contains := by
  rw [SortedNatTree.contains_eq_list row3761 row3761_ordered, row3761_values]
def row3767 : SortedNatTree.Tree := (.node 12041 (.node 6911 (.node 4937 (.node 4127 (.node 3947 (.node 3821 .empty .empty) .empty) (.node 4523 .empty .empty)) (.node 6701 (.node 6521 (.node 5987 .empty .empty) .empty) (.node 6719 .empty .empty))) (.node 7793 (.node 7487 (.node 7193 (.node 7001 .empty .empty) .empty) (.node 7547 .empty .empty)) (.node 9437 (.node 9377 (.node 8933 .empty .empty) .empty) (.node 11717 .empty .empty)))) (.node 17597 (.node 14087 (.node 13751 (.node 13499 (.node 12113 .empty .empty) .empty) (.node 13763 .empty .empty)) (.node 17333 (.node 15803 (.node 14771 .empty .empty) .empty) (.node 17417 .empty .empty))) (.node 19697 (.node 19373 (.node 19121 (.node 18341 .empty .empty) .empty) (.node 19583 .empty .empty)) (.node 19793 (.node 19727 .empty .empty) (.node 21701 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3767_ordered : SortedNatTree.ordered row3767 = true := by decide
set_option maxRecDepth 4096 in
theorem row3767_values : SortedNatTree.values row3767 = Data.row3767 := rfl
theorem row3767_checked : SortedNatTree.contains row3767 = Data.row3767.contains := by
  rw [SortedNatTree.contains_eq_list row3767 row3767_ordered, row3767_values]
def row3769 : SortedNatTree.Tree := (.node 14779 (.node 7669 (.node 5653 (.node 4597 (.node 4051 .empty .empty) (.node 4603 .empty .empty)) (.node 6949 (.node 6619 .empty .empty) .empty)) (.node 10789 (.node 8971 (.node 8161 .empty .empty) (.node 9127 .empty .empty)) (.node 13219 (.node 12589 .empty .empty) .empty))) (.node 19687 (.node 16747 (.node 15307 (.node 15139 .empty .empty) (.node 16063 .empty .empty)) (.node 18379 (.node 16993 .empty .empty) .empty)) (.node 21313 (.node 19927 (.node 19843 .empty .empty) .empty) (.node 22027 (.node 21577 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3769_ordered : SortedNatTree.ordered row3769 = true := by decide
set_option maxRecDepth 4096 in
theorem row3769_values : SortedNatTree.values row3769 = Data.row3769 := rfl
theorem row3769_checked : SortedNatTree.contains row3769 = Data.row3769.contains := by
  rw [SortedNatTree.contains_eq_list row3769 row3769_ordered, row3769_values]
def row3779 : SortedNatTree.Tree := (.node 12671 (.node 6329 (.node 5099 (.node 4637 (.node 4241 (.node 3863 .empty .empty) .empty) (.node 4931 .empty .empty)) (.node 6053 (.node 5843 .empty .empty) (.node 6299 .empty .empty))) (.node 8573 (.node 7577 (.node 7523 .empty .empty) (.node 7829 .empty .empty)) (.node 10529 (.node 9941 .empty .empty) (.node 11927 .empty .empty)))) (.node 14951 (.node 14447 (.node 13241 (.node 12911 .empty .empty) (.node 13259 .empty .empty)) (.node 14561 (.node 14549 .empty .empty) (.node 14897 .empty .empty))) (.node 18539 (.node 17783 (.node 17489 .empty .empty) (.node 18059 .empty .empty)) (.node 21347 (.node 19727 .empty .empty) (.node 21893 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3779_ordered : SortedNatTree.ordered row3779 = true := by decide
set_option maxRecDepth 4096 in
theorem row3779_values : SortedNatTree.values row3779 = Data.row3779 := rfl
theorem row3779_checked : SortedNatTree.contains row3779 = Data.row3779.contains := by
  rw [SortedNatTree.contains_eq_list row3779 row3779_ordered, row3779_values]
def row3793 : SortedNatTree.Tree := (.node 11731 (.node 7873 (.node 6907 (.node 5659 (.node 5179 .empty .empty) (.node 6733 .empty .empty)) (.node 7213 (.node 6949 .empty .empty) .empty)) (.node 10093 (.node 8179 (.node 8167 .empty .empty) (.node 9007 .empty .empty)) (.node 11689 (.node 10753 .empty .empty) .empty))) (.node 18859 (.node 16363 (.node 14011 (.node 13219 .empty .empty) (.node 15439 .empty .empty)) (.node 17293 (.node 16981 .empty .empty) .empty)) (.node 20611 (.node 20353 (.node 18913 .empty .empty) .empty) (.node 21277 (.node 21169 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3793_ordered : SortedNatTree.ordered row3793 = true := by decide
set_option maxRecDepth 4096 in
theorem row3793_values : SortedNatTree.values row3793 = Data.row3793 := rfl
theorem row3793_checked : SortedNatTree.contains row3793 = Data.row3793.contains := by
  rw [SortedNatTree.contains_eq_list row3793 row3793_ordered, row3793_values]
def row3797 : SortedNatTree.Tree := (.node 12917 (.node 7937 (.node 6089 (.node 5261 (.node 4049 (.node 3833 .empty .empty) .empty) (.node 5393 (.node 5303 .empty .empty) .empty)) (.node 7109 (.node 7019 (.node 7013 .empty .empty) .empty) (.node 7529 .empty .empty))) (.node 10271 (.node 8861 (.node 8537 (.node 8273 .empty .empty) .empty) (.node 9839 (.node 9791 .empty .empty) .empty)) (.node 11399 (.node 10889 (.node 10733 .empty .empty) .empty) (.node 12413 .empty .empty)))) (.node 16229 (.node 14891 (.node 13883 (.node 13487 (.node 13331 .empty .empty) .empty) (.node 14033 (.node 13967 .empty .empty) .empty)) (.node 15923 (.node 15641 (.node 15263 .empty .empty) .empty) (.node 16193 .empty .empty))) (.node 19763 (.node 18587 (.node 18371 (.node 17981 .empty .empty) .empty) (.node 19709 (.node 19073 .empty .empty) .empty)) (.node 20807 (.node 20261 (.node 20123 .empty .empty) .empty) (.node 21851 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3797_ordered : SortedNatTree.ordered row3797 = true := by decide
set_option maxRecDepth 4096 in
theorem row3797_values : SortedNatTree.values row3797 = Data.row3797 := rfl
theorem row3797_checked : SortedNatTree.contains row3797 = Data.row3797.contains := by
  rw [SortedNatTree.contains_eq_list row3797 row3797_ordered, row3797_values]
def row3803 : SortedNatTree.Tree := (.node 13127 (.node 7253 (.node 5051 (.node 4289 (.node 4001 (.node 3911 .empty .empty) .empty) (.node 4943 (.node 4547 .empty .empty) .empty)) (.node 6521 (.node 5507 (.node 5087 .empty .empty) .empty) (.node 7121 .empty .empty))) (.node 11171 (.node 9587 (.node 8999 (.node 8069 .empty .empty) .empty) (.node 10949 (.node 9941 .empty .empty) .empty)) (.node 12539 (.node 11801 (.node 11213 .empty .empty) .empty) (.node 12671 .empty .empty)))) (.node 17159 (.node 15083 (.node 13421 (.node 13259 (.node 13151 .empty .empty) .empty) (.node 13931 (.node 13523 .empty .empty) .empty)) (.node 16127 (.node 16073 (.node 15443 .empty .empty) .empty) (.node 16253 .empty .empty))) (.node 18773 (.node 18311 (.node 17939 (.node 17291 .empty .empty) .empty) (.node 18503 .empty .empty)) (.node 20747 (.node 20147 (.node 19991 .empty .empty) .empty) (.node 21419 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3803_ordered : SortedNatTree.ordered row3803 = true := by decide
set_option maxRecDepth 4096 in
theorem row3803_values : SortedNatTree.values row3803 = Data.row3803 := rfl
theorem row3803_checked : SortedNatTree.contains row3803 = Data.row3803.contains := by
  rw [SortedNatTree.contains_eq_list row3803 row3803_ordered, row3803_values]
def row3821 : SortedNatTree.Tree := (.node 13241 (.node 8933 (.node 6863 (.node 6263 (.node 6047 (.node 4703 .empty .empty) .empty) (.node 6473 .empty .empty)) (.node 7019 (.node 6917 .empty .empty) (.node 7307 .empty .empty))) (.node 10067 (.node 9539 (.node 8951 .empty .empty) (.node 9803 .empty .empty)) (.node 12821 (.node 11399 .empty .empty) (.node 13187 .empty .empty)))) (.node 16937 (.node 14669 (.node 14633 (.node 14591 (.node 13679 .empty .empty) .empty) (.node 14657 .empty .empty)) (.node 15497 (.node 14867 .empty .empty) (.node 16451 .empty .empty))) (.node 18917 (.node 17627 (.node 17609 .empty .empty) (.node 18371 .empty .empty)) (.node 20117 (.node 19403 .empty .empty) (.node 21101 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3821_ordered : SortedNatTree.ordered row3821 = true := by decide
set_option maxRecDepth 4096 in
theorem row3821_values : SortedNatTree.values row3821 = Data.row3821 := rfl
theorem row3821_checked : SortedNatTree.contains row3821 = Data.row3821.contains := by
  rw [SortedNatTree.contains_eq_list row3821 row3821_ordered, row3821_values]
def row3823 : SortedNatTree.Tree := (.node 11131 (.node 6199 (.node 4759 (.node 4297 (.node 4201 (.node 4159 .empty .empty) .empty) (.node 4549 .empty .empty)) (.node 5233 (.node 5209 (.node 5167 .empty .empty) .empty) (.node 5779 .empty .empty))) (.node 8689 (.node 7417 (.node 7393 (.node 6841 .empty .empty) .empty) (.node 7993 .empty .empty)) (.node 10597 (.node 10069 (.node 9181 .empty .empty) .empty) (.node 10711 .empty .empty)))) (.node 15859 (.node 13093 (.node 12163 (.node 11833 (.node 11617 .empty .empty) .empty) (.node 12517 .empty .empty)) (.node 14869 (.node 14563 (.node 13831 .empty .empty) .empty) (.node 14923 .empty .empty))) (.node 19489 (.node 18787 (.node 18427 (.node 16369 .empty .empty) .empty) (.node 19237 .empty .empty)) (.node 21001 (.node 20899 (.node 20269 .empty .empty) .empty) (.node 21559 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3823_ordered : SortedNatTree.ordered row3823 = true := by decide
set_option maxRecDepth 4096 in
theorem row3823_values : SortedNatTree.values row3823 = Data.row3823 := rfl
theorem row3823_checked : SortedNatTree.contains row3823 = Data.row3823.contains := by
  rw [SortedNatTree.contains_eq_list row3823 row3823_ordered, row3823_values]
def row3833 : SortedNatTree.Tree := (.node 13001 (.node 6833 (.node 5813 (.node 4889 (.node 4733 (.node 4253 .empty .empty) .empty) (.node 5657 .empty .empty)) (.node 6449 (.node 6269 .empty .empty) (.node 6827 .empty .empty))) (.node 10799 (.node 7841 (.node 6983 (.node 6911 .empty .empty) .empty) (.node 9857 .empty .empty)) (.node 11447 (.node 11177 .empty .empty) (.node 11657 .empty .empty)))) (.node 18539 (.node 14891 (.node 14621 (.node 13913 (.node 13829 .empty .empty) .empty) (.node 14753 .empty .empty)) (.node 15767 (.node 15629 .empty .empty) (.node 18371 .empty .empty))) (.node 21149 (.node 19457 (.node 18743 .empty .empty) (.node 19793 .empty .empty)) (.node 22067 (.node 21383 .empty .empty) (.node 22073 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3833_ordered : SortedNatTree.ordered row3833 = true := by decide
set_option maxRecDepth 4096 in
theorem row3833_values : SortedNatTree.values row3833 = Data.row3833 := rfl
theorem row3833_checked : SortedNatTree.contains row3833 = Data.row3833.contains := by
  rw [SortedNatTree.contains_eq_list row3833 row3833_ordered, row3833_values]
def row3847 : SortedNatTree.Tree := (.node 10141 (.node 6883 (.node 5659 (.node 4099 (.node 4021 (.node 3931 .empty .empty) .empty) (.node 5197 (.node 5077 .empty .empty) .empty)) (.node 6367 (.node 6211 (.node 5737 .empty .empty) .empty) (.node 6553 .empty .empty))) (.node 8929 (.node 8821 (.node 8233 (.node 8221 .empty .empty) .empty) (.node 8887 .empty .empty)) (.node 9949 (.node 9901 (.node 9817 .empty .empty) .empty) (.node 10069 .empty .empty)))) (.node 17041 (.node 13183 (.node 12049 (.node 11863 (.node 10789 .empty .empty) .empty) (.node 12967 (.node 12451 .empty .empty) .empty)) (.node 16369 (.node 15277 (.node 13267 .empty .empty) .empty) (.node 16567 .empty .empty))) (.node 21061 (.node 19507 (.node 19273 (.node 17239 .empty .empty) .empty) (.node 19681 .empty .empty)) (.node 21613 (.node 21319 (.node 21157 .empty .empty) .empty) (.node 21751 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3847_ordered : SortedNatTree.ordered row3847 = true := by decide
set_option maxRecDepth 4096 in
theorem row3847_values : SortedNatTree.values row3847 = Data.row3847 := rfl
theorem row3847_checked : SortedNatTree.contains row3847 = Data.row3847.contains := by
  rw [SortedNatTree.contains_eq_list row3847 row3847_ordered, row3847_values]
def row3851 : SortedNatTree.Tree := (.node 11717 (.node 7043 (.node 5903 (.node 4271 (.node 4139 (.node 4013 .empty .empty) .empty) (.node 5651 (.node 4649 .empty .empty) .empty)) (.node 6899 (.node 6659 (.node 6521 .empty .empty) .empty) (.node 6983 .empty .empty))) (.node 9491 (.node 7673 (.node 7547 (.node 7151 .empty .empty) .empty) (.node 8663 .empty .empty)) (.node 10133 (.node 9839 (.node 9521 .empty .empty) .empty) (.node 10181 .empty .empty)))) (.node 16811 (.node 13037 (.node 12743 (.node 12041 (.node 11981 .empty .empty) .empty) (.node 12821 .empty .empty)) (.node 16661 (.node 15233 (.node 13877 .empty .empty) .empty) (.node 16703 .empty .empty))) (.node 18911 (.node 17921 (.node 17891 (.node 17657 .empty .empty) .empty) (.node 18593 .empty .empty)) (.node 20051 (.node 19991 (.node 19463 .empty .empty) .empty) (.node 21557 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3851_ordered : SortedNatTree.ordered row3851 = true := by decide
set_option maxRecDepth 4096 in
theorem row3851_values : SortedNatTree.values row3851 = Data.row3851 := rfl
theorem row3851_checked : SortedNatTree.contains row3851 = Data.row3851.contains := by
  rw [SortedNatTree.contains_eq_list row3851 row3851_ordered, row3851_values]
def row3853 : SortedNatTree.Tree := (.node 8923 (.node 6991 (.node 6361 (.node 5521 (.node 4759 (.node 4099 .empty .empty) .empty) (.node 5683 .empty .empty)) (.node 6607 (.node 6367 .empty .empty) (.node 6709 .empty .empty))) (.node 7699 (.node 7297 (.node 7243 (.node 7237 .empty .empty) .empty) (.node 7591 .empty .empty)) (.node 8389 (.node 8293 .empty .empty) (.node 8887 .empty .empty)))) (.node 14107 (.node 10273 (.node 9433 (.node 9349 (.node 9103 .empty .empty) .empty) (.node 10111 .empty .empty)) (.node 11119 (.node 10399 .empty .empty) (.node 13297 .empty .empty))) (.node 19219 (.node 18457 (.node 16273 (.node 15787 .empty .empty) .empty) (.node 18979 .empty .empty)) (.node 21487 (.node 21061 .empty .empty) (.node 21499 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3853_ordered : SortedNatTree.ordered row3853 = true := by decide
set_option maxRecDepth 4096 in
theorem row3853_values : SortedNatTree.values row3853 = Data.row3853 := rfl
theorem row3853_checked : SortedNatTree.contains row3853 = Data.row3853.contains := by
  rw [SortedNatTree.contains_eq_list row3853 row3853_ordered, row3853_values]
def row3863 : SortedNatTree.Tree := (.node 12323 (.node 8543 (.node 6779 (.node 5297 (.node 4451 (.node 4007 .empty .empty) .empty) (.node 6569 .empty .empty)) (.node 7673 (.node 7187 .empty .empty) (.node 7757 .empty .empty))) (.node 10739 (.node 8867 (.node 8849 .empty .empty) (.node 9203 .empty .empty)) (.node 11789 (.node 11777 .empty .empty) (.node 12041 .empty .empty)))) (.node 19289 (.node 17471 (.node 16979 (.node 14879 (.node 14717 .empty .empty) .empty) (.node 17093 .empty .empty)) (.node 18719 (.node 18329 .empty .empty) (.node 19013 .empty .empty))) (.node 21017 (.node 20201 (.node 19463 .empty .empty) (.node 20261 .empty .empty)) (.node 21977 (.node 21323 .empty .empty) (.node 22037 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3863_ordered : SortedNatTree.ordered row3863 = true := by decide
set_option maxRecDepth 4096 in
theorem row3863_values : SortedNatTree.values row3863 = Data.row3863 := rfl
theorem row3863_checked : SortedNatTree.contains row3863 = Data.row3863.contains := by
  rw [SortedNatTree.contains_eq_list row3863 row3863_ordered, row3863_values]
def row3877 : SortedNatTree.Tree := (.node 10501 (.node 6163 (.node 5479 (.node 5167 (.node 4507 .empty .empty) (.node 5227 .empty .empty)) (.node 6151 (.node 5683 .empty .empty) .empty)) (.node 7159 (.node 6469 (.node 6373 .empty .empty) (.node 6607 .empty .empty)) (.node 9049 (.node 8263 .empty .empty) .empty))) (.node 19219 (.node 18049 (.node 15217 (.node 12781 .empty .empty) (.node 17431 .empty .empty)) (.node 18679 (.node 18553 .empty .empty) .empty)) (.node 21013 (.node 19543 (.node 19471 .empty .empty) (.node 20731 .empty .empty)) (.node 21799 (.node 21487 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3877_ordered : SortedNatTree.ordered row3877 = true := by decide
set_option maxRecDepth 4096 in
theorem row3877_values : SortedNatTree.values row3877 = Data.row3877 := rfl
theorem row3877_checked : SortedNatTree.contains row3877 = Data.row3877.contains := by
  rw [SortedNatTree.contains_eq_list row3877 row3877_ordered, row3877_values]
def row3881 : SortedNatTree.Tree := (.node 12899 (.node 8369 (.node 5519 (.node 4793 (.node 4421 .empty .empty) (.node 5261 .empty .empty)) (.node 6857 (.node 6269 .empty .empty) (.node 8363 .empty .empty))) (.node 9419 (.node 8867 (.node 8609 .empty .empty) (.node 9281 .empty .empty)) (.node 10589 (.node 10253 .empty .empty) (.node 12401 .empty .empty)))) (.node 15971 (.node 14591 (.node 14411 (.node 12953 .empty .empty) (.node 14537 .empty .empty)) (.node 15053 (.node 14633 .empty .empty) (.node 15263 .empty .empty))) (.node 20921 (.node 18869 (.node 18701 .empty .empty) (.node 20549 .empty .empty)) (.node 21557 (.node 21317 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3881_ordered : SortedNatTree.ordered row3881 = true := by decide
set_option maxRecDepth 4096 in
theorem row3881_values : SortedNatTree.values row3881 = Data.row3881 := rfl
theorem row3881_checked : SortedNatTree.contains row3881 = Data.row3881.contains := by
  rw [SortedNatTree.contains_eq_list row3881 row3881_ordered, row3881_values]
def row3889 : SortedNatTree.Tree := (.node 13729 (.node 8647 (.node 6619 (.node 5821 (.node 5749 (.node 4021 .empty .empty) .empty) (.node 6007 .empty .empty)) (.node 7477 (.node 6763 .empty .empty) (.node 7927 .empty .empty))) (.node 11503 (.node 9439 (.node 9103 (.node 8677 .empty .empty) .empty) (.node 9781 .empty .empty)) (.node 12973 (.node 12889 .empty .empty) (.node 13597 .empty .empty)))) (.node 17827 (.node 16069 (.node 14947 (.node 14887 (.node 13807 .empty .empty) .empty) (.node 15859 .empty .empty)) (.node 17257 (.node 17047 .empty .empty) (.node 17377 .empty .empty))) (.node 19813 (.node 19309 (.node 18553 (.node 18253 .empty .empty) .empty) (.node 19603 .empty .empty)) (.node 20011 (.node 19819 .empty .empty) (.node 21751 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3889_ordered : SortedNatTree.ordered row3889 = true := by decide
set_option maxRecDepth 4096 in
theorem row3889_values : SortedNatTree.values row3889 = Data.row3889 := rfl
theorem row3889_checked : SortedNatTree.contains row3889 = Data.row3889.contains := by
  rw [SortedNatTree.contains_eq_list row3889 row3889_ordered, row3889_values]
def row3907 : SortedNatTree.Tree := (.node 13003 (.node 6997 (.node 5431 (.node 4327 (.node 4027 .empty .empty) (.node 4603 .empty .empty)) (.node 6469 (.node 6043 .empty .empty) (.node 6871 .empty .empty))) (.node 9619 (.node 7573 (.node 7333 .empty .empty) (.node 8887 .empty .empty)) (.node 12919 (.node 12601 .empty .empty) .empty))) (.node 19051 (.node 16561 (.node 15427 (.node 14503 .empty .empty) (.node 15991 .empty .empty)) (.node 18061 (.node 16747 .empty .empty) .empty)) (.node 20593 (.node 20149 (.node 19867 .empty .empty) (.node 20287 .empty .empty)) (.node 21589 (.node 21157 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3907_ordered : SortedNatTree.ordered row3907 = true := by decide
set_option maxRecDepth 4096 in
theorem row3907_values : SortedNatTree.values row3907 = Data.row3907 := rfl
theorem row3907_checked : SortedNatTree.contains row3907 = Data.row3907.contains := by
  rw [SortedNatTree.contains_eq_list row3907 row3907_ordered, row3907_values]
def row3911 : SortedNatTree.Tree := (.node 10739 (.node 7937 (.node 5717 (.node 4457 (.node 4259 (.node 4079 .empty .empty) .empty) (.node 5477 .empty .empty)) (.node 7349 (.node 6257 .empty .empty) (.node 7919 .empty .empty))) (.node 9539 (.node 9257 (.node 8867 .empty .empty) (.node 9497 .empty .empty)) (.node 9851 (.node 9791 .empty .empty) (.node 10463 .empty .empty)))) (.node 14957 (.node 13523 (.node 12983 (.node 12269 (.node 11579 .empty .empty) .empty) (.node 13127 .empty .empty)) (.node 14519 (.node 14303 .empty .empty) (.node 14741 .empty .empty))) (.node 18461 (.node 16889 (.node 15233 .empty .empty) (.node 17807 .empty .empty)) (.node 20903 (.node 18773 .empty .empty) (.node 21491 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3911_ordered : SortedNatTree.ordered row3911 = true := by decide
set_option maxRecDepth 4096 in
theorem row3911_values : SortedNatTree.values row3911 = Data.row3911 := rfl
theorem row3911_checked : SortedNatTree.contains row3911 = Data.row3911.contains := by
  rw [SortedNatTree.contains_eq_list row3911 row3911_ordered, row3911_values]
def row3917 : SortedNatTree.Tree := (.node 8543 (.node 5399 (.node 4337 (.node 4211 (.node 4019 .empty .empty) (.node 4271 .empty .empty)) (.node 4649 (.node 4349 .empty .empty) (.node 5189 .empty .empty))) (.node 6779 (.node 5987 (.node 5483 .empty .empty) (.node 6323 .empty .empty)) (.node 7331 (.node 6899 .empty .empty) (.node 7499 .empty .empty)))) (.node 14249 (.node 12329 (.node 10181 (.node 8753 .empty .empty) (.node 12107 .empty .empty)) (.node 13163 (.node 12791 .empty .empty) (.node 13457 .empty .empty))) (.node 18443 (.node 17231 (.node 15671 .empty .empty) (.node 17609 .empty .empty)) (.node 20183 (.node 18869 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3917_ordered : SortedNatTree.ordered row3917 = true := by decide
set_option maxRecDepth 4096 in
theorem row3917_values : SortedNatTree.values row3917 = Data.row3917 := rfl
theorem row3917_checked : SortedNatTree.contains row3917 = Data.row3917.contains := by
  rw [SortedNatTree.contains_eq_list row3917 row3917_ordered, row3917_values]
def row3919 : SortedNatTree.Tree := (.node 9103 (.node 6151 (.node 5179 (.node 4591 (.node 4567 (.node 4297 .empty .empty) .empty) (.node 4957 .empty .empty)) (.node 5743 (.node 5503 (.node 5281 .empty .empty) .empty) (.node 6007 .empty .empty))) (.node 7489 (.node 6961 (.node 6529 (.node 6397 .empty .empty) .empty) (.node 7309 .empty .empty)) (.node 8011 (.node 7789 .empty .empty) (.node 8317 .empty .empty)))) (.node 17599 (.node 13723 (.node 11677 (.node 9547 (.node 9133 .empty .empty) .empty) (.node 13339 .empty .empty)) (.node 14251 (.node 13831 .empty .empty) (.node 16141 .empty .empty))) (.node 20173 (.node 20101 (.node 19993 (.node 17749 .empty .empty) .empty) (.node 20143 .empty .empty)) (.node 20773 (.node 20719 .empty .empty) (.node 22039 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3919_ordered : SortedNatTree.ordered row3919 = true := by decide
set_option maxRecDepth 4096 in
theorem row3919_values : SortedNatTree.values row3919 = Data.row3919 := rfl
theorem row3919_checked : SortedNatTree.contains row3919 = Data.row3919.contains := by
  rw [SortedNatTree.contains_eq_list row3919 row3919_ordered, row3919_values]
def row3923 : SortedNatTree.Tree := (.node 9041 (.node 6011 (.node 4721 (.node 4211 (.node 4133 (.node 3989 .empty .empty) .empty) (.node 4703 .empty .empty)) (.node 4973 (.node 4919 .empty .empty) (.node 5519 .empty .empty))) (.node 8627 (.node 7949 (.node 7643 (.node 6803 .empty .empty) .empty) (.node 8237 .empty .empty)) (.node 8783 (.node 8663 .empty .empty) (.node 8861 .empty .empty)))) (.node 14411 (.node 11399 (.node 10313 (.node 9467 (.node 9377 .empty .empty) .empty) (.node 10937 .empty .empty)) (.node 12497 (.node 11783 .empty .empty) (.node 13241 .empty .empty))) (.node 19583 (.node 19373 (.node 18233 (.node 16103 .empty .empty) .empty) (.node 19427 .empty .empty)) (.node 20297 (.node 19997 .empty .empty) (.node 20771 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3923_ordered : SortedNatTree.ordered row3923 = true := by decide
set_option maxRecDepth 4096 in
theorem row3923_values : SortedNatTree.values row3923 = Data.row3923 := rfl
theorem row3923_checked : SortedNatTree.contains row3923 = Data.row3923.contains := by
  rw [SortedNatTree.contains_eq_list row3923 row3923_ordered, row3923_values]
def row3929 : SortedNatTree.Tree := (.node 12437 (.node 7691 (.node 6221 (.node 6053 (.node 4421 (.node 4349 .empty .empty) .empty) (.node 6143 .empty .empty)) (.node 7109 (.node 7043 .empty .empty) (.node 7457 .empty .empty))) (.node 10487 (.node 9521 (.node 9209 (.node 9059 .empty .empty) .empty) (.node 9941 .empty .empty)) (.node 11447 (.node 11171 .empty .empty) (.node 11549 .empty .empty)))) (.node 17417 (.node 15401 (.node 14723 (.node 13799 (.node 13049 .empty .empty) .empty) (.node 14891 .empty .empty)) (.node 16319 (.node 15737 .empty .empty) (.node 17387 .empty .empty))) (.node 19433 (.node 19073 (.node 17747 (.node 17597 .empty .empty) .empty) (.node 19139 .empty .empty)) (.node 20393 (.node 19913 .empty .empty) (.node 21929 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3929_ordered : SortedNatTree.ordered row3929 = true := by decide
set_option maxRecDepth 4096 in
theorem row3929_values : SortedNatTree.values row3929 = Data.row3929 := rfl
theorem row3929_checked : SortedNatTree.contains row3929 = Data.row3929.contains := by
  rw [SortedNatTree.contains_eq_list row3929 row3929_ordered, row3929_values]
def row3931 : SortedNatTree.Tree := (.node 15601 (.node 7639 (.node 6661 (.node 5527 (.node 5449 (.node 5413 .empty .empty) .empty) (.node 6553 .empty .empty)) (.node 7507 (.node 7459 .empty .empty) (.node 7561 .empty .empty))) (.node 9403 (.node 8233 (.node 8089 .empty .empty) (.node 8803 .empty .empty)) (.node 12589 (.node 9619 .empty .empty) (.node 13681 .empty .empty)))) (.node 17971 (.node 16747 (.node 16189 (.node 16033 .empty .empty) (.node 16333 .empty .empty)) (.node 17029 (.node 16963 .empty .empty) (.node 17419 .empty .empty))) (.node 20011 (.node 19471 (.node 19087 .empty .empty) (.node 19681 .empty .empty)) (.node 21031 (.node 20149 .empty .empty) (.node 21397 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3931_ordered : SortedNatTree.ordered row3931 = true := by decide
set_option maxRecDepth 4096 in
theorem row3931_values : SortedNatTree.values row3931 = Data.row3931 := rfl
theorem row3931_checked : SortedNatTree.contains row3931 = Data.row3931.contains := by
  rw [SortedNatTree.contains_eq_list row3931 row3931_ordered, row3931_values]
def row3943 : SortedNatTree.Tree := (.node 12619 (.node 8059 (.node 5581 (.node 5413 (.node 5167 (.node 4621 .empty .empty) .empty) (.node 5569 (.node 5479 .empty .empty) .empty)) (.node 6829 (.node 5839 (.node 5779 .empty .empty) .empty) (.node 7789 .empty .empty))) (.node 9769 (.node 8923 (.node 8647 (.node 8563 .empty .empty) .empty) (.node 9277 .empty .empty)) (.node 12277 (.node 10627 (.node 9787 .empty .empty) .empty) (.node 12553 .empty .empty)))) (.node 16927 (.node 14887 (.node 14083 (.node 13873 (.node 12979 .empty .empty) .empty) (.node 14593 (.node 14533 .empty .empty) .empty)) (.node 15661 (.node 15199 (.node 15139 .empty .empty) .empty) (.node 16183 .empty .empty))) (.node 19507 (.node 18523 (.node 18367 (.node 17239 .empty .empty) .empty) (.node 19213 .empty .empty)) (.node 21727 (.node 20431 (.node 19753 .empty .empty) .empty) (.node 22003 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3943_ordered : SortedNatTree.ordered row3943 = true := by decide
set_option maxRecDepth 4096 in
theorem row3943_values : SortedNatTree.values row3943 = Data.row3943 := rfl
theorem row3943_checked : SortedNatTree.contains row3943 = Data.row3943.contains := by
  rw [SortedNatTree.contains_eq_list row3943 row3943_ordered, row3943_values]
def row3947 : SortedNatTree.Tree := (.node 11813 (.node 6803 (.node 5393 (.node 4889 (.node 4679 (.node 4073 .empty .empty) .empty) (.node 5171 .empty .empty)) (.node 6263 (.node 6011 (.node 5801 .empty .empty) .empty) (.node 6653 .empty .empty))) (.node 8837 (.node 8147 (.node 8093 (.node 7001 .empty .empty) .empty) (.node 8513 .empty .empty)) (.node 9173 (.node 8861 (.node 8849 .empty .empty) .empty) (.node 9857 .empty .empty)))) (.node 16103 (.node 14057 (.node 12713 (.node 12347 (.node 11897 .empty .empty) .empty) (.node 12743 .empty .empty)) (.node 15137 (.node 14741 (.node 14081 .empty .empty) .empty) (.node 15803 .empty .empty))) (.node 18959 (.node 17909 (.node 17627 (.node 17333 .empty .empty) .empty) (.node 18917 .empty .empty)) (.node 20789 (.node 20681 (.node 20441 .empty .empty) .empty) (.node 21773 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3947_ordered : SortedNatTree.ordered row3947 = true := by decide
set_option maxRecDepth 4096 in
theorem row3947_values : SortedNatTree.values row3947 = Data.row3947 := rfl
theorem row3947_checked : SortedNatTree.contains row3947 = Data.row3947.contains := by
  rw [SortedNatTree.contains_eq_list row3947 row3947_ordered, row3947_values]
def row3967 : SortedNatTree.Tree := (.node 11173 (.node 7159 (.node 5953 (.node 5011 (.node 4783 (.node 4177 .empty .empty) .empty) (.node 5791 .empty .empty)) (.node 6211 (.node 6067 .empty .empty) (.node 6271 .empty .empty))) (.node 8053 (.node 7927 (.node 7699 (.node 7177 .empty .empty) .empty) (.node 7933 .empty .empty)) (.node 8893 (.node 8209 .empty .empty) (.node 9781 .empty .empty)))) (.node 17851 (.node 13627 (.node 12517 (.node 11971 (.node 11467 .empty .empty) .empty) (.node 13249 .empty .empty)) (.node 16189 (.node 15973 .empty .empty) (.node 16381 .empty .empty))) (.node 20011 (.node 18661 (.node 18313 .empty .empty) (.node 19777 .empty .empty)) (.node 21187 (.node 21001 .empty .empty) (.node 21649 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3967_ordered : SortedNatTree.ordered row3967 = true := by decide
set_option maxRecDepth 4096 in
theorem row3967_values : SortedNatTree.values row3967 = Data.row3967 := rfl
theorem row3967_checked : SortedNatTree.contains row3967 = Data.row3967.contains := by
  rw [SortedNatTree.contains_eq_list row3967 row3967_ordered, row3967_values]
def row3989 : SortedNatTree.Tree := (.node 9623 (.node 8093 (.node 6581 (.node 5783 (.node 4931 (.node 4157 .empty .empty) .empty) (.node 5927 .empty .empty)) (.node 7043 (.node 6719 .empty .empty) (.node 7703 .empty .empty))) (.node 8933 (.node 8597 (.node 8513 (.node 8291 .empty .empty) .empty) (.node 8861 .empty .empty)) (.node 9311 (.node 9239 .empty .empty) (.node 9491 .empty .empty)))) (.node 17573 (.node 13901 (.node 11633 (.node 10133 (.node 10037 .empty .empty) .empty) (.node 13469 .empty .empty)) (.node 14759 (.node 14207 .empty .empty) (.node 15791 .empty .empty))) (.node 20231 (.node 19403 (.node 19289 (.node 17921 .empty .empty) .empty) (.node 20177 .empty .empty)) (.node 20411 (.node 20261 .empty .empty) (.node 21491 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3989_ordered : SortedNatTree.ordered row3989 = true := by decide
set_option maxRecDepth 4096 in
theorem row3989_values : SortedNatTree.values row3989 = Data.row3989 := rfl
theorem row3989_checked : SortedNatTree.contains row3989 = Data.row3989.contains := by
  rw [SortedNatTree.contains_eq_list row3989 row3989_ordered, row3989_values]
def row4001 : SortedNatTree.Tree := (.node 12329 (.node 7577 (.node 6323 (.node 4391 (.node 4349 (.node 4253 .empty .empty) .empty) (.node 5783 .empty .empty)) (.node 6563 (.node 6329 .empty .empty) (.node 7121 .empty .empty))) (.node 8543 (.node 8219 (.node 7877 (.node 7583 .empty .empty) .empty) (.node 8513 .empty .empty)) (.node 9689 (.node 9467 .empty .empty) (.node 10253 .empty .empty)))) (.node 16889 (.node 15629 (.node 13877 (.node 13619 (.node 12569 .empty .empty) .empty) (.node 14723 .empty .empty)) (.node 16091 (.node 15773 .empty .empty) (.node 16319 .empty .empty))) (.node 20759 (.node 19919 (.node 17609 .empty .empty) (.node 19997 .empty .empty)) (.node 21089 (.node 20807 .empty .empty) (.node 22031 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4001_ordered : SortedNatTree.ordered row4001 = true := by decide
set_option maxRecDepth 4096 in
theorem row4001_values : SortedNatTree.values row4001 = Data.row4001 := rfl
theorem row4001_checked : SortedNatTree.contains row4001 = Data.row4001.contains := by
  rw [SortedNatTree.contains_eq_list row4001 row4001_ordered, row4001_values]
def row4003 : SortedNatTree.Tree := (.node 11953 (.node 8539 (.node 5503 (.node 4723 (.node 4597 (.node 4567 .empty .empty) .empty) (.node 5431 .empty .empty)) (.node 6163 (.node 5851 .empty .empty) (.node 8527 .empty .empty))) (.node 9511 (.node 8839 (.node 8629 .empty .empty) (.node 9049 .empty .empty)) (.node 10303 (.node 9787 .empty .empty) (.node 11161 .empty .empty)))) (.node 18199 (.node 14503 (.node 12967 (.node 12547 (.node 12097 .empty .empty) .empty) (.node 13063 .empty .empty)) (.node 15373 (.node 15259 .empty .empty) (.node 16057 .empty .empty))) (.node 18973 (.node 18439 (.node 18229 .empty .empty) (.node 18637 .empty .empty)) (.node 19681 (.node 19477 .empty .empty) (.node 19753 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4003_ordered : SortedNatTree.ordered row4003 = true := by decide
set_option maxRecDepth 4096 in
theorem row4003_values : SortedNatTree.values row4003 = Data.row4003 := rfl
theorem row4003_checked : SortedNatTree.contains row4003 = Data.row4003.contains := by
  rw [SortedNatTree.contains_eq_list row4003 row4003_ordered, row4003_values]
def row4007 : SortedNatTree.Tree := (.node 11321 (.node 6353 (.node 5417 (.node 4337 (.node 4217 (.node 4091 .empty .empty) .empty) (.node 4751 .empty .empty)) (.node 6011 (.node 5639 .empty .empty) (.node 6053 .empty .empty))) (.node 8243 (.node 7907 (.node 7853 (.node 7283 .empty .empty) .empty) (.node 8081 .empty .empty)) (.node 8693 (.node 8369 .empty .empty) (.node 10883 .empty .empty)))) (.node 17579 (.node 16091 (.node 13487 (.node 12149 (.node 12113 .empty .empty) .empty) (.node 15173 .empty .empty)) (.node 17117 (.node 16553 .empty .empty) (.node 17519 .empty .empty))) (.node 19997 (.node 18233 (.node 17909 .empty .empty) (.node 19211 .empty .empty)) (.node 21599 (.node 20759 .empty .empty) (.node 21767 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4007_ordered : SortedNatTree.ordered row4007 = true := by decide
set_option maxRecDepth 4096 in
theorem row4007_values : SortedNatTree.values row4007 = Data.row4007 := rfl
theorem row4007_checked : SortedNatTree.contains row4007 = Data.row4007.contains := by
  rw [SortedNatTree.contains_eq_list row4007 row4007_ordered, row4007_values]
def row4013 : SortedNatTree.Tree := (.node 12203 (.node 7541 (.node 5573 (.node 4799 (.node 4643 (.node 4421 .empty .empty) .empty) (.node 5153 .empty .empty)) (.node 5867 (.node 5741 .empty .empty) (.node 6659 .empty .empty))) (.node 8867 (.node 8387 (.node 8069 .empty .empty) (.node 8597 .empty .empty)) (.node 11549 (.node 10037 .empty .empty) (.node 11909 .empty .empty)))) (.node 17627 (.node 14723 (.node 13313 (.node 12539 (.node 12479 .empty .empty) .empty) (.node 13967 .empty .empty)) (.node 16427 (.node 16193 .empty .empty) (.node 17387 .empty .empty))) (.node 20393 (.node 19403 (.node 17957 .empty .empty) (.node 19973 .empty .empty)) (.node 20753 (.node 20483 .empty .empty) (.node 20903 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4013_ordered : SortedNatTree.ordered row4013 = true := by decide
set_option maxRecDepth 4096 in
theorem row4013_values : SortedNatTree.values row4013 = Data.row4013 := rfl
theorem row4013_checked : SortedNatTree.contains row4013 = Data.row4013.contains := by
  rw [SortedNatTree.contains_eq_list row4013 row4013_ordered, row4013_values]
def row4019 : SortedNatTree.Tree := (.node 10433 (.node 7541 (.node 6791 (.node 5471 (.node 4211 .empty .empty) .empty) (.node 7211 (.node 7121 .empty .empty) .empty)) (.node 10139 (.node 9941 (.node 9479 .empty .empty) .empty) (.node 10181 .empty .empty))) (.node 18719 (.node 15161 (.node 13901 (.node 11987 .empty .empty) .empty) (.node 17903 (.node 17489 .empty .empty) .empty)) (.node 20231 (.node 20063 (.node 19571 .empty .empty) .empty) (.node 20903 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row4019_ordered : SortedNatTree.ordered row4019 = true := by decide
set_option maxRecDepth 4096 in
theorem row4019_values : SortedNatTree.values row4019 = Data.row4019 := rfl
theorem row4019_checked : SortedNatTree.contains row4019 = Data.row4019.contains := by
  rw [SortedNatTree.contains_eq_list row4019 row4019_ordered, row4019_values]
def row4021 : SortedNatTree.Tree := (.node 13417 (.node 9349 (.node 5563 (.node 4903 (.node 4441 .empty .empty) (.node 5347 .empty .empty)) (.node 7207 (.node 6733 .empty .empty) (.node 7537 .empty .empty))) (.node 11923 (.node 10453 (.node 9547 .empty .empty) (.node 10987 .empty .empty)) (.node 12043 (.node 12037 .empty .empty) (.node 12163 .empty .empty)))) (.node 17923 (.node 15313 (.node 14143 (.node 13879 .empty .empty) (.node 14737 .empty .empty)) (.node 16831 (.node 16651 .empty .empty) (.node 16921 .empty .empty))) (.node 20857 (.node 18097 (.node 17929 .empty .empty) (.node 18973 .empty .empty)) (.node 21523 (.node 20899 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4021_ordered : SortedNatTree.ordered row4021 = true := by decide
set_option maxRecDepth 4096 in
theorem row4021_values : SortedNatTree.values row4021 = Data.row4021 := rfl
theorem row4021_checked : SortedNatTree.contains row4021 = Data.row4021.contains := by
  rw [SortedNatTree.contains_eq_list row4021 row4021_ordered, row4021_values]
def row4027 : SortedNatTree.Tree := (.node 12577 (.node 7417 (.node 5503 (.node 5233 (.node 4933 (.node 4783 .empty .empty) .empty) (.node 5413 .empty .empty)) (.node 6871 (.node 6199 .empty .empty) (.node 6949 .empty .empty))) (.node 9181 (.node 8863 (.node 8761 (.node 8677 .empty .empty) .empty) (.node 9157 .empty .empty)) (.node 11443 (.node 9277 .empty .empty) (.node 12109 .empty .empty)))) (.node 15541 (.node 14419 (.node 13789 (.node 13669 (.node 13537 .empty .empty) .empty) (.node 14083 .empty .empty)) (.node 14929 (.node 14461 .empty .empty) (.node 15427 .empty .empty))) (.node 18181 (.node 16879 (.node 16759 (.node 15727 .empty .empty) .empty) (.node 16903 .empty .empty)) (.node 19717 (.node 18793 .empty .empty) (.node 20389 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4027_ordered : SortedNatTree.ordered row4027 = true := by decide
set_option maxRecDepth 4096 in
theorem row4027_values : SortedNatTree.values row4027 = Data.row4027 := rfl
theorem row4027_checked : SortedNatTree.contains row4027 = Data.row4027.contains := by
  rw [SortedNatTree.contains_eq_list row4027 row4027_ordered, row4027_values]
def row4049 : SortedNatTree.Tree := (.node 12899 (.node 10139 (.node 5927 (.node 4919 (.node 4877 (.node 4133 .empty .empty) .empty) (.node 5261 .empty .empty)) (.node 8111 (.node 7703 (.node 7649 .empty .empty) .empty) (.node 9623 .empty .empty))) (.node 12329 (.node 11939 (.node 11519 (.node 11279 .empty .empty) .empty) (.node 12323 .empty .empty)) (.node 12689 (.node 12653 .empty .empty) (.node 12791 .empty .empty)))) (.node 17351 (.node 14303 (.node 13967 (.node 13877 (.node 13151 .empty .empty) .empty) (.node 14249 .empty .empty)) (.node 16649 (.node 14621 .empty .empty) (.node 17099 .empty .empty))) (.node 19763 (.node 19289 (.node 19259 (.node 17393 .empty .empty) .empty) (.node 19751 .empty .empty)) (.node 20627 (.node 20249 .empty .empty) (.node 21467 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4049_ordered : SortedNatTree.ordered row4049 = true := by decide
set_option maxRecDepth 4096 in
theorem row4049_values : SortedNatTree.values row4049 = Data.row4049 := rfl
theorem row4049_checked : SortedNatTree.contains row4049 = Data.row4049.contains := by
  rw [SortedNatTree.contains_eq_list row4049 row4049_ordered, row4049_values]
def row4051 : SortedNatTree.Tree := (.node 10957 (.node 7549 (.node 6121 (.node 5077 (.node 4657 (.node 4129 .empty .empty) .empty) (.node 5701 (.node 5521 .empty .empty) .empty)) (.node 7027 (.node 6841 (.node 6211 .empty .empty) .empty) (.node 7219 .empty .empty))) (.node 8563 (.node 8263 (.node 7951 (.node 7591 .empty .empty) .empty) (.node 8311 .empty .empty)) (.node 10729 (.node 9133 (.node 8689 .empty .empty) .empty) (.node 10789 .empty .empty)))) (.node 17431 (.node 14173 (.node 12613 (.node 12451 (.node 12157 .empty .empty) .empty) (.node 13873 .empty .empty)) (.node 15913 (.node 15259 (.node 15013 .empty .empty) .empty) (.node 16141 .empty .empty))) (.node 20161 (.node 19609 (.node 18457 (.node 18289 .empty .empty) .empty) (.node 20113 .empty .empty)) (.node 21397 (.node 21187 (.node 20887 .empty .empty) .empty) (.node 21481 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4051_ordered : SortedNatTree.ordered row4051 = true := by decide
set_option maxRecDepth 4096 in
theorem row4051_values : SortedNatTree.values row4051 = Data.row4051 := rfl
theorem row4051_checked : SortedNatTree.contains row4051 = Data.row4051.contains := by
  rw [SortedNatTree.contains_eq_list row4051 row4051_ordered, row4051_values]
def row4057 : SortedNatTree.Tree := (.node 12919 (.node 7393 (.node 6007 (.node 5077 (.node 4663 (.node 4483 .empty .empty) .empty) (.node 5419 .empty .empty)) (.node 6469 (.node 6421 (.node 6199 .empty .empty) .empty) (.node 7213 .empty .empty))) (.node 9109 (.node 8821 (.node 7879 (.node 7753 .empty .empty) .empty) (.node 9103 .empty .empty)) (.node 12583 (.node 11083 (.node 9829 .empty .empty) .empty) (.node 12889 .empty .empty)))) (.node 19759 (.node 14737 (.node 14347 (.node 14107 (.node 13597 .empty .empty) .empty) (.node 14653 .empty .empty)) (.node 17881 (.node 17497 (.node 15733 .empty .empty) .empty) (.node 18541 .empty .empty))) (.node 20857 (.node 20611 (.node 20533 (.node 19927 .empty .empty) .empty) (.node 20743 .empty .empty)) (.node 21523 (.node 21313 (.node 21163 .empty .empty) .empty) (.node 21751 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4057_ordered : SortedNatTree.ordered row4057 = true := by decide
set_option maxRecDepth 4096 in
theorem row4057_values : SortedNatTree.values row4057 = Data.row4057 := rfl
theorem row4057_checked : SortedNatTree.contains row4057 = Data.row4057.contains := by
  rw [SortedNatTree.contains_eq_list row4057 row4057_ordered, row4057_values]
def row4073 : SortedNatTree.Tree := (.node 13553 (.node 8573 (.node 5393 (.node 5003 (.node 4877 (.node 4481 .empty .empty) .empty) (.node 5087 .empty .empty)) (.node 6977 (.node 6389 .empty .empty) (.node 7901 .empty .empty))) (.node 9539 (.node 9029 (.node 8861 (.node 8627 .empty .empty) .empty) (.node 9341 .empty .empty)) (.node 13007 (.node 10301 .empty .empty) (.node 13037 .empty .empty)))) (.node 17123 (.node 15761 (.node 14969 (.node 14939 (.node 13763 .empty .empty) .empty) (.node 15383 .empty .empty)) (.node 16829 (.node 16493 .empty .empty) (.node 16901 .empty .empty))) (.node 19697 (.node 18731 (.node 17837 (.node 17627 .empty .empty) .empty) (.node 19079 .empty .empty)) (.node 20261 (.node 20063 .empty .empty) (.node 21017 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4073_ordered : SortedNatTree.ordered row4073 = true := by decide
set_option maxRecDepth 4096 in
theorem row4073_values : SortedNatTree.values row4073 = Data.row4073 := rfl
theorem row4073_checked : SortedNatTree.contains row4073 = Data.row4073.contains := by
  rw [SortedNatTree.contains_eq_list row4073 row4073_ordered, row4073_values]
def row4079 : SortedNatTree.Tree := (.node 9743 (.node 5927 (.node 5441 (.node 5231 (.node 4973 .empty .empty) (.node 5309 .empty .empty)) (.node 5897 (.node 5879 .empty .empty) .empty)) (.node 6131 (.node 6053 (.node 6029 .empty .empty) .empty) (.node 9323 (.node 6863 .empty .empty) .empty))) (.node 18047 (.node 13151 (.node 11069 (.node 10103 .empty .empty) (.node 12203 .empty .empty)) (.node 14537 (.node 13553 .empty .empty) .empty)) (.node 20147 (.node 19997 (.node 19661 .empty .empty) .empty) (.node 21383 (.node 21149 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4079_ordered : SortedNatTree.ordered row4079 = true := by decide
set_option maxRecDepth 4096 in
theorem row4079_values : SortedNatTree.values row4079 = Data.row4079 := rfl
theorem row4079_checked : SortedNatTree.contains row4079 = Data.row4079.contains := by
  rw [SortedNatTree.contains_eq_list row4079 row4079_ordered, row4079_values]
def row4091 : SortedNatTree.Tree := (.node 10973 (.node 7577 (.node 5669 (.node 4283 (.node 4241 .empty .empty) (.node 4919 .empty .empty)) (.node 7103 (.node 6173 .empty .empty) (.node 7559 .empty .empty))) (.node 9491 (.node 8237 (.node 7937 .empty .empty) (.node 9161 .empty .empty)) (.node 10313 (.node 9839 .empty .empty) .empty))) (.node 14549 (.node 13841 (.node 12941 (.node 11321 .empty .empty) (.node 13241 .empty .empty)) (.node 14423 (.node 14369 .empty .empty) .empty)) (.node 19793 (.node 18539 (.node 16937 .empty .empty) (.node 19751 .empty .empty)) (.node 20753 (.node 19973 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4091_ordered : SortedNatTree.ordered row4091 = true := by decide
set_option maxRecDepth 4096 in
theorem row4091_values : SortedNatTree.values row4091 = Data.row4091 := rfl
theorem row4091_checked : SortedNatTree.contains row4091 = Data.row4091.contains := by
  rw [SortedNatTree.contains_eq_list row4091 row4091_ordered, row4091_values]
def row4093 : SortedNatTree.Tree := (.node 11503 (.node 7321 (.node 5659 (.node 5077 (.node 4567 .empty .empty) (.node 5119 .empty .empty)) (.node 6823 (.node 5749 .empty .empty) (.node 6991 .empty .empty))) (.node 9403 (.node 8167 (.node 7933 .empty .empty) (.node 8647 .empty .empty)) (.node 11173 (.node 9721 .empty .empty) (.node 11467 .empty .empty)))) (.node 13687 (.node 12391 (.node 11971 (.node 11587 .empty .empty) (.node 12109 .empty .empty)) (.node 12703 (.node 12601 .empty .empty) (.node 13183 .empty .empty))) (.node 16057 (.node 15139 (.node 14449 .empty .empty) (.node 15877 .empty .empty)) (.node 16417 (.node 16189 .empty .empty) (.node 18181 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4093_ordered : SortedNatTree.ordered row4093 = true := by decide
set_option maxRecDepth 4096 in
theorem row4093_values : SortedNatTree.values row4093 = Data.row4093 := rfl
theorem row4093_checked : SortedNatTree.contains row4093 = Data.row4093.contains := by
  rw [SortedNatTree.contains_eq_list row4093 row4093_ordered, row4093_values]
def row4099 : SortedNatTree.Tree := (.node 14221 (.node 10909 (.node 5953 (.node 4909 (.node 4783 .empty .empty) (.node 4999 .empty .empty)) (.node 8821 (.node 6793 .empty .empty) .empty)) (.node 13099 (.node 12553 (.node 12157 .empty .empty) (.node 12757 .empty .empty)) (.node 13597 (.node 13267 .empty .empty) .empty))) (.node 16693 (.node 15787 (.node 15061 (.node 14551 .empty .empty) (.node 15439 .empty .empty)) (.node 15901 (.node 15817 .empty .empty) .empty)) (.node 19141 (.node 18133 (.node 16993 .empty .empty) (.node 18211 .empty .empty)) (.node 21397 (.node 19219 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4099_ordered : SortedNatTree.ordered row4099 = true := by decide
set_option maxRecDepth 4096 in
theorem row4099_values : SortedNatTree.values row4099 = Data.row4099 := rfl
theorem row4099_checked : SortedNatTree.contains row4099 = Data.row4099.contains := by
  rw [SortedNatTree.contains_eq_list row4099 row4099_ordered, row4099_values]
def row4111 : SortedNatTree.Tree := (.node 11353 (.node 7369 (.node 6073 (.node 5521 (.node 5119 (.node 4261 .empty .empty) .empty) (.node 5857 (.node 5527 .empty .empty) .empty)) (.node 6661 (.node 6637 (.node 6553 .empty .empty) .empty) (.node 6949 (.node 6679 .empty .empty) .empty))) (.node 9811 (.node 8719 (.node 8167 (.node 7789 .empty .empty) .empty) (.node 9349 (.node 9181 .empty .empty) .empty)) (.node 10639 (.node 10627 (.node 10597 .empty .empty) .empty) (.node 10957 .empty .empty)))) (.node 17359 (.node 13267 (.node 12157 (.node 12073 (.node 11593 .empty .empty) .empty) (.node 13249 (.node 12391 .empty .empty) .empty)) (.node 15877 (.node 14887 (.node 14551 .empty .empty) .empty) (.node 16417 .empty .empty))) (.node 19843 (.node 18457 (.node 17467 (.node 17401 .empty .empty) .empty) (.node 19699 (.node 18919 .empty .empty) .empty)) (.node 21019 (.node 21001 (.node 20341 .empty .empty) .empty) (.node 21817 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4111_ordered : SortedNatTree.ordered row4111 = true := by decide
set_option maxRecDepth 4096 in
theorem row4111_values : SortedNatTree.values row4111 = Data.row4111 := rfl
theorem row4111_checked : SortedNatTree.contains row4111 = Data.row4111.contains := by
  rw [SortedNatTree.contains_eq_list row4111 row4111_ordered, row4111_values]
def row4127 : SortedNatTree.Tree := (.node 12653 (.node 8243 (.node 6737 (.node 5519 (.node 5309 (.node 4967 .empty .empty) .empty) (.node 6581 .empty .empty)) (.node 7079 (.node 6959 .empty .empty) (.node 7949 .empty .empty))) (.node 9551 (.node 8837 (.node 8291 .empty .empty) (.node 8933 .empty .empty)) (.node 11681 (.node 9767 .empty .empty) (.node 12203 .empty .empty)))) (.node 17489 (.node 13751 (.node 12983 (.node 12953 (.node 12917 .empty .empty) .empty) (.node 13001 .empty .empty)) (.node 15551 (.node 14957 .empty .empty) (.node 17117 .empty .empty))) (.node 19913 (.node 18539 (.node 18443 .empty .empty) (.node 19841 .empty .empty)) (.node 20879 (.node 20807 .empty .empty) (.node 20981 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4127_ordered : SortedNatTree.ordered row4127 = true := by decide
set_option maxRecDepth 4096 in
theorem row4127_values : SortedNatTree.values row4127 = Data.row4127 := rfl
theorem row4127_checked : SortedNatTree.contains row4127 = Data.row4127.contains := by
  rw [SortedNatTree.contains_eq_list row4127 row4127_ordered, row4127_values]
def row4129 : SortedNatTree.Tree := (.node 12373 (.node 8221 (.node 6781 (.node 5521 (.node 4861 (.node 4651 .empty .empty) .empty) (.node 6361 (.node 6271 .empty .empty) .empty)) (.node 7717 (.node 7159 (.node 6949 .empty .empty) .empty) (.node 8167 .empty .empty))) (.node 10723 (.node 9343 (.node 9091 (.node 8377 .empty .empty) .empty) (.node 9397 .empty .empty)) (.node 11353 (.node 11131 (.node 11071 .empty .empty) .empty) (.node 11833 .empty .empty)))) (.node 17317 (.node 14533 (.node 13003 (.node 12757 (.node 12457 .empty .empty) .empty) (.node 14431 .empty .empty)) (.node 17107 (.node 15991 (.node 15859 .empty .empty) .empty) (.node 17203 .empty .empty))) (.node 19471 (.node 18061 (.node 18043 (.node 17389 .empty .empty) .empty) (.node 18439 .empty .empty)) (.node 21559 (.node 21031 (.node 20857 .empty .empty) .empty) (.node 21841 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4129_ordered : SortedNatTree.ordered row4129 = true := by decide
set_option maxRecDepth 4096 in
theorem row4129_values : SortedNatTree.values row4129 = Data.row4129 := rfl
theorem row4129_checked : SortedNatTree.contains row4129 = Data.row4129.contains := by
  rw [SortedNatTree.contains_eq_list row4129 row4129_ordered, row4129_values]
def row4133 : SortedNatTree.Tree := (.node 12569 (.node 7433 (.node 5171 (.node 4721 (.node 4493 (.node 4409 .empty .empty) .empty) (.node 5081 .empty .empty)) (.node 6173 (.node 5387 (.node 5351 .empty .empty) .empty) (.node 6947 .empty .empty))) (.node 9371 (.node 9173 (.node 8111 (.node 7451 .empty .empty) .empty) (.node 9293 .empty .empty)) (.node 12269 (.node 10169 .empty .empty) (.node 12329 .empty .empty)))) (.node 17747 (.node 15647 (.node 14009 (.node 13001 (.node 12953 .empty .empty) .empty) (.node 14051 .empty .empty)) (.node 17231 (.node 16061 (.node 15803 .empty .empty) .empty) (.node 17579 .empty .empty))) (.node 21023 (.node 20849 (.node 19763 (.node 19157 .empty .empty) .empty) (.node 21011 .empty .empty)) (.node 21647 (.node 21179 .empty .empty) (.node 21863 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4133_ordered : SortedNatTree.ordered row4133 = true := by decide
set_option maxRecDepth 4096 in
theorem row4133_values : SortedNatTree.values row4133 = Data.row4133 := rfl
theorem row4133_checked : SortedNatTree.contains row4133 = Data.row4133.contains := by
  rw [SortedNatTree.contains_eq_list row4133 row4133_ordered, row4133_values]
def row4139 : SortedNatTree.Tree := (.node 10499 (.node 7607 (.node 6959 (.node 6263 (.node 5657 (.node 4937 .empty .empty) .empty) (.node 6911 .empty .empty)) (.node 7457 (.node 7253 .empty .empty) (.node 7541 .empty .empty))) (.node 8783 (.node 8369 (.node 7673 .empty .empty) (.node 8387 .empty .empty)) (.node 8969 (.node 8867 .empty .empty) (.node 9749 .empty .empty)))) (.node 15443 (.node 12641 (.node 11177 (.node 11159 .empty .empty) (.node 11423 .empty .empty)) (.node 13691 (.node 13103 .empty .empty) (.node 15233 .empty .empty))) (.node 17393 (.node 16001 (.node 15959 .empty .empty) (.node 16253 .empty .empty)) (.node 20693 (.node 18899 .empty .empty) (.node 21713 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4139_ordered : SortedNatTree.ordered row4139 = true := by decide
set_option maxRecDepth 4096 in
theorem row4139_values : SortedNatTree.values row4139 = Data.row4139 := rfl
theorem row4139_checked : SortedNatTree.contains row4139 = Data.row4139.contains := by
  rw [SortedNatTree.contains_eq_list row4139 row4139_ordered, row4139_values]
def row4153 : SortedNatTree.Tree := (.node 11353 (.node 6997 (.node 5077 (.node 4789 (.node 4201 .empty .empty) (.node 5023 .empty .empty)) (.node 5503 (.node 5101 .empty .empty) (.node 5521 .empty .empty))) (.node 9283 (.node 8311 (.node 7753 .empty .empty) (.node 8821 .empty .empty)) (.node 10837 (.node 9661 .empty .empty) .empty))) (.node 15289 (.node 12553 (.node 12343 (.node 11593 .empty .empty) (.node 12409 .empty .empty)) (.node 14869 (.node 14071 .empty .empty) .empty)) (.node 18481 (.node 17971 (.node 15913 .empty .empty) (.node 18181 .empty .empty)) (.node 20809 (.node 19753 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4153_ordered : SortedNatTree.ordered row4153 = true := by decide
set_option maxRecDepth 4096 in
theorem row4153_values : SortedNatTree.values row4153 = Data.row4153 := rfl
theorem row4153_checked : SortedNatTree.contains row4153 = Data.row4153.contains := by
  rw [SortedNatTree.contains_eq_list row4153 row4153_ordered, row4153_values]
def row4157 : SortedNatTree.Tree := (.node 12953 (.node 7433 (.node 6947 (.node 5717 (.node 5333 .empty .empty) .empty) (.node 7127 (.node 7043 .empty .empty) .empty)) (.node 9497 (.node 8741 (.node 7577 .empty .empty) .empty) (.node 11159 (.node 10259 .empty .empty) .empty))) (.node 19211 (.node 15803 (.node 14771 (.node 14657 .empty .empty) .empty) (.node 17807 (.node 16547 .empty .empty) .empty)) (.node 21599 (.node 21221 (.node 20693 .empty .empty) .empty) (.node 21773 (.node 21701 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4157_ordered : SortedNatTree.ordered row4157 = true := by decide
set_option maxRecDepth 4096 in
theorem row4157_values : SortedNatTree.values row4157 = Data.row4157 := rfl
theorem row4157_checked : SortedNatTree.contains row4157 = Data.row4157.contains := by
  rw [SortedNatTree.contains_eq_list row4157 row4157_ordered, row4157_values]
def row4159 : SortedNatTree.Tree := (.node 13033 (.node 8233 (.node 5779 (.node 4789 (.node 4621 .empty .empty) (.node 5011 .empty .empty)) (.node 7573 (.node 6661 .empty .empty) (.node 7681 .empty .empty))) (.node 11161 (.node 9769 (.node 9643 .empty .empty) (.node 10069 .empty .empty)) (.node 12037 (.node 11677 .empty .empty) (.node 12163 .empty .empty)))) (.node 18049 (.node 15541 (.node 14767 (.node 13513 .empty .empty) (.node 15121 .empty .empty)) (.node 17209 (.node 16759 .empty .empty) (.node 17839 .empty .empty))) (.node 19429 (.node 18757 (.node 18397 .empty .empty) (.node 19417 .empty .empty)) (.node 21067 (.node 19681 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4159_ordered : SortedNatTree.ordered row4159 = true := by decide
set_option maxRecDepth 4096 in
theorem row4159_values : SortedNatTree.values row4159 = Data.row4159 := rfl
theorem row4159_checked : SortedNatTree.contains row4159 = Data.row4159.contains := by
  rw [SortedNatTree.contains_eq_list row4159 row4159_ordered, row4159_values]
def row4177 : SortedNatTree.Tree := (.node 8923 (.node 6883 (.node 5791 (.node 4903 (.node 4363 .empty .empty) (.node 5323 .empty .empty)) (.node 6673 (.node 6379 .empty .empty) (.node 6781 .empty .empty))) (.node 8101 (.node 7537 (.node 7297 .empty .empty) (.node 8053 .empty .empty)) (.node 8353 (.node 8311 .empty .empty) (.node 8521 .empty .empty)))) (.node 15919 (.node 11587 (.node 9973 (.node 9781 .empty .empty) (.node 10303 .empty .empty)) (.node 13693 (.node 12487 .empty .empty) (.node 13963 .empty .empty))) (.node 17107 (.node 16183 (.node 15937 .empty .empty) (.node 16417 .empty .empty)) (.node 20593 (.node 19441 .empty .empty) (.node 21169 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4177_ordered : SortedNatTree.ordered row4177 = true := by decide
set_option maxRecDepth 4096 in
theorem row4177_values : SortedNatTree.values row4177 = Data.row4177 := rfl
theorem row4177_checked : SortedNatTree.contains row4177 = Data.row4177.contains := by
  rw [SortedNatTree.contains_eq_list row4177 row4177_ordered, row4177_values]
def row4201 : SortedNatTree.Tree := (.node 10753 (.node 8161 (.node 5749 (.node 5419 (.node 4759 .empty .empty) (.node 5527 .empty .empty)) (.node 6217 (.node 6199 .empty .empty) (.node 6421 .empty .empty))) (.node 9403 (.node 8971 (.node 8863 .empty .empty) (.node 9067 .empty .empty)) (.node 10597 (.node 9421 .empty .empty) .empty))) (.node 16339 (.node 14281 (.node 12379 (.node 11437 .empty .empty) (.node 13219 .empty .empty)) (.node 15877 (.node 14797 .empty .empty) (.node 15991 .empty .empty))) (.node 19237 (.node 17041 (.node 16729 .empty .empty) (.node 19069 .empty .empty)) (.node 21529 (.node 21379 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4201_ordered : SortedNatTree.ordered row4201 = true := by decide
set_option maxRecDepth 4096 in
theorem row4201_values : SortedNatTree.values row4201 = Data.row4201 := rfl
theorem row4201_checked : SortedNatTree.contains row4201 = Data.row4201.contains := by
  rw [SortedNatTree.contains_eq_list row4201 row4201_ordered, row4201_values]
def row4211 : SortedNatTree.Tree := (.node 13619 (.node 9221 (.node 5693 (.node 5351 (.node 5237 .empty .empty) (.node 5483 .empty .empty)) (.node 7703 (.node 6911 .empty .empty) (.node 8963 .empty .empty))) (.node 10607 (.node 10181 (.node 9887 .empty .empty) (.node 10601 .empty .empty)) (.node 12227 (.node 10973 .empty .empty) .empty))) (.node 18671 (.node 15803 (.node 14669 (.node 14033 .empty .empty) (.node 14717 .empty .empty)) (.node 18041 (.node 16811 .empty .empty) .empty)) (.node 21227 (.node 19763 (.node 19289 .empty .empty) (.node 19973 .empty .empty)) (.node 21737 (.node 21647 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4211_ordered : SortedNatTree.ordered row4211 = true := by decide
set_option maxRecDepth 4096 in
theorem row4211_values : SortedNatTree.values row4211 = Data.row4211 := rfl
theorem row4211_checked : SortedNatTree.contains row4211 = Data.row4211.contains := by
  rw [SortedNatTree.contains_eq_list row4211 row4211_ordered, row4211_values]
def row4217 : SortedNatTree.Tree := (.node 11177 (.node 7829 (.node 4751 (.node 4421 (.node 4337 .empty .empty) (.node 4673 .empty .empty)) (.node 5981 (.node 5171 .empty .empty) (.node 7823 .empty .empty))) (.node 9377 (.node 8363 (.node 8111 .empty .empty) (.node 9209 .empty .empty)) (.node 9719 (.node 9629 .empty .empty) (.node 9743 .empty .empty)))) (.node 14537 (.node 11597 (.node 11393 (.node 11261 .empty .empty) (.node 11489 .empty .empty)) (.node 12479 (.node 12413 .empty .empty) (.node 14327 .empty .empty))) (.node 20549 (.node 15641 (.node 15467 .empty .empty) (.node 17909 .empty .empty)) (.node 20879 (.node 20771 .empty .empty) (.node 21401 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4217_ordered : SortedNatTree.ordered row4217 = true := by decide
set_option maxRecDepth 4096 in
theorem row4217_values : SortedNatTree.values row4217 = Data.row4217 := rfl
theorem row4217_checked : SortedNatTree.contains row4217 = Data.row4217.contains := by
  rw [SortedNatTree.contains_eq_list row4217 row4217_ordered, row4217_values]
def row4219 : SortedNatTree.Tree := (.node 9403 (.node 6379 (.node 5227 (.node 5113 (.node 4597 .empty .empty) (.node 5119 .empty .empty)) (.node 5749 (.node 5521 .empty .empty) (.node 6337 .empty .empty))) (.node 7027 (.node 6793 (.node 6529 .empty .empty) (.node 6883 .empty .empty)) (.node 8209 (.node 7369 .empty .empty) (.node 8707 .empty .empty)))) (.node 15679 (.node 12889 (.node 10141 (.node 9679 .empty .empty) (.node 10771 .empty .empty)) (.node 14197 (.node 13669 .empty .empty) (.node 14797 .empty .empty))) (.node 20743 (.node 18493 (.node 15937 .empty .empty) (.node 18583 .empty .empty)) (.node 21601 (.node 21013 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4219_ordered : SortedNatTree.ordered row4219 = true := by decide
set_option maxRecDepth 4096 in
theorem row4219_values : SortedNatTree.values row4219 = Data.row4219 := rfl
theorem row4219_checked : SortedNatTree.contains row4219 = Data.row4219.contains := by
  rw [SortedNatTree.contains_eq_list row4219 row4219_ordered, row4219_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
