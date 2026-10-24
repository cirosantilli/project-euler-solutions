import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row1637 : SortedNatTree.Tree := (.node 11777 (.node 6053 (.node 4073 (.node 2903 (.node 1889 (.node 1709 .empty .empty) .empty) (.node 3911 (.node 3359 .empty .empty) .empty)) (.node 4973 (.node 4931 (.node 4649 .empty .empty) .empty) (.node 5897 (.node 5669 .empty .empty) .empty))) (.node 8237 (.node 7841 (.node 7607 (.node 7013 .empty .empty) .empty) (.node 8039 (.node 8009 .empty .empty) .empty)) (.node 9479 (.node 8933 (.node 8363 .empty .empty) .empty) (.node 10667 (.node 9833 .empty .empty) .empty)))) (.node 19289 (.node 15107 (.node 14633 (.node 13397 (.node 12347 .empty .empty) .empty) (.node 14741 (.node 14669 .empty .empty) .empty)) (.node 16481 (.node 15467 (.node 15329 .empty .empty) .empty) (.node 18539 (.node 16937 .empty .empty) .empty))) (.node 22571 (.node 20483 (.node 20327 (.node 20261 .empty .empty) .empty) (.node 22271 (.node 21851 .empty .empty) .empty)) (.node 23297 (.node 22877 (.node 22859 .empty .empty) .empty) (.node 23561 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1637_ordered : SortedNatTree.ordered row1637 = true := by decide
set_option maxRecDepth 4096 in
theorem row1637_values : SortedNatTree.values row1637 = Data.row1637 := rfl
theorem row1637_checked : SortedNatTree.contains row1637 = Data.row1637.contains := by
  rw [SortedNatTree.contains_eq_list row1637 row1637_ordered, row1637_values]
def row1657 : SortedNatTree.Tree := (.node 9931 (.node 4969 (.node 2917 (.node 2683 (.node 1879 (.node 1699 .empty .empty) (.node 2203 .empty .empty)) (.node 2887 (.node 2791 .empty .empty) .empty)) (.node 4111 (.node 3967 (.node 3463 .empty .empty) .empty) (.node 4549 (.node 4273 .empty .empty) .empty))) (.node 8017 (.node 6733 (.node 5101 (.node 5059 .empty .empty) (.node 6397 .empty .empty)) (.node 7237 (.node 6907 .empty .empty) .empty)) (.node 8623 (.node 8269 (.node 8167 .empty .empty) .empty) (.node 9817 (.node 8647 .empty .empty) .empty)))) (.node 17029 (.node 12739 (.node 10789 (.node 10627 (.node 10513 .empty .empty) (.node 10639 .empty .empty)) (.node 12301 (.node 11047 .empty .empty) .empty)) (.node 14947 (.node 14449 (.node 13903 .empty .empty) .empty) (.node 16693 (.node 15787 .empty .empty) .empty))) (.node 21433 (.node 18793 (.node 18199 (.node 17239 .empty .empty) (.node 18481 .empty .empty)) (.node 20809 (.node 20173 .empty .empty) .empty)) (.node 23473 (.node 22699 (.node 22549 .empty .empty) .empty) (.node 23977 (.node 23563 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1657_ordered : SortedNatTree.ordered row1657 = true := by decide
set_option maxRecDepth 4096 in
theorem row1657_values : SortedNatTree.values row1657 = Data.row1657 := rfl
theorem row1657_checked : SortedNatTree.contains row1657 = Data.row1657.contains := by
  rw [SortedNatTree.contains_eq_list row1657 row1657_ordered, row1657_values]
def row1663 : SortedNatTree.Tree := (.node 11083 (.node 7879 (.node 4987 (.node 4027 (.node 3229 (.node 2857 .empty .empty) (.node 3613 .empty .empty)) (.node 4831 (.node 4231 .empty .empty) .empty)) (.node 6007 (.node 5449 (.node 5413 .empty .empty) (.node 5653 .empty .empty)) (.node 7219 (.node 6577 .empty .empty) .empty))) (.node 9619 (.node 9067 (.node 8221 (.node 7963 .empty .empty) (.node 8443 .empty .empty)) (.node 9391 (.node 9349 .empty .empty) .empty)) (.node 10267 (.node 9871 (.node 9769 .empty .empty) .empty) (.node 10909 (.node 10831 .empty .empty) .empty)))) (.node 16231 (.node 13597 (.node 12889 (.node 12409 (.node 11923 .empty .empty) (.node 12583 .empty .empty)) (.node 13339 (.node 12973 .empty .empty) .empty)) (.node 14143 (.node 13687 (.node 13681 .empty .empty) (.node 14107 .empty .empty)) (.node 15901 (.node 15859 .empty .empty) .empty))) (.node 20443 (.node 18583 (.node 16417 (.node 16273 .empty .empty) (.node 16963 .empty .empty)) (.node 20347 (.node 20071 .empty .empty) .empty)) (.node 22381 (.node 21577 (.node 20899 .empty .empty) .empty) (.node 23371 (.node 23293 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1663_ordered : SortedNatTree.ordered row1663 = true := by decide
set_option maxRecDepth 4096 in
theorem row1663_values : SortedNatTree.values row1663 = Data.row1663 := rfl
theorem row1663_checked : SortedNatTree.contains row1663 = Data.row1663.contains := by
  rw [SortedNatTree.contains_eq_list row1663 row1663_ordered, row1663_values]
def row1667 : SortedNatTree.Tree := (.node 10559 (.node 5051 (.node 3929 (.node 2267 (.node 1973 (.node 1901 .empty .empty) (.node 2153 .empty .empty)) (.node 3623 (.node 2657 .empty .empty) .empty)) (.node 4703 (.node 4481 (.node 4157 .empty .empty) .empty) (.node 4967 (.node 4817 .empty .empty) .empty))) (.node 6353 (.node 6287 (.node 5573 (.node 5531 .empty .empty) (.node 5669 .empty .empty)) (.node 6329 (.node 6323 .empty .empty) .empty)) (.node 9137 (.node 8663 (.node 6701 .empty .empty) .empty) (.node 10259 (.node 10091 .empty .empty) .empty)))) (.node 19181 (.node 16631 (.node 14447 (.node 13457 (.node 12113 .empty .empty) (.node 14321 .empty .empty)) (.node 16349 (.node 15233 .empty .empty) .empty)) (.node 17987 (.node 16883 (.node 16787 .empty .empty) .empty) (.node 18701 (.node 18059 .empty .empty) .empty))) (.node 22091 (.node 21701 (.node 20639 (.node 20129 .empty .empty) .empty) (.node 21881 (.node 21773 .empty .empty) .empty)) (.node 23741 (.node 23663 (.node 22877 .empty .empty) .empty) (.node 23873 (.node 23747 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1667_ordered : SortedNatTree.ordered row1667 = true := by decide
set_option maxRecDepth 4096 in
theorem row1667_values : SortedNatTree.values row1667 = Data.row1667 := rfl
theorem row1667_checked : SortedNatTree.contains row1667 = Data.row1667.contains := by
  rw [SortedNatTree.contains_eq_list row1667 row1667_ordered, row1667_values]
def row1669 : SortedNatTree.Tree := (.node 9733 (.node 5743 (.node 3499 (.node 2791 (.node 2707 (.node 2593 .empty .empty) .empty) (.node 3301 (.node 3181 .empty .empty) .empty)) (.node 4441 (.node 4273 (.node 3511 .empty .empty) .empty) (.node 5659 (.node 4651 .empty .empty) .empty))) (.node 7867 (.node 6709 (.node 6661 (.node 6577 .empty .empty) .empty) (.node 7669 (.node 6733 .empty .empty) .empty)) (.node 8839 (.node 8599 (.node 7873 .empty .empty) .empty) (.node 9091 .empty .empty)))) (.node 18379 (.node 13381 (.node 11959 (.node 11497 (.node 10729 .empty .empty) .empty) (.node 13099 (.node 12457 .empty .empty) .empty)) (.node 17341 (.node 16831 (.node 16447 .empty .empty) .empty) (.node 17749 .empty .empty))) (.node 19603 (.node 18979 (.node 18919 (.node 18679 .empty .empty) .empty) (.node 19483 (.node 19213 .empty .empty) .empty)) (.node 22153 (.node 21163 (.node 20029 .empty .empty) .empty) (.node 24061 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1669_ordered : SortedNatTree.ordered row1669 = true := by decide
set_option maxRecDepth 4096 in
theorem row1669_values : SortedNatTree.values row1669 = Data.row1669 := rfl
theorem row1669_checked : SortedNatTree.contains row1669 = Data.row1669.contains := by
  rw [SortedNatTree.contains_eq_list row1669 row1669_ordered, row1669_values]
def row1693 : SortedNatTree.Tree := (.node 12967 (.node 6217 (.node 3373 (.node 2593 (.node 2371 (.node 1753 .empty .empty) (.node 2551 .empty .empty)) (.node 3181 (.node 2689 .empty .empty) .empty)) (.node 4327 (.node 3919 (.node 3739 .empty .empty) .empty) (.node 4639 (.node 4441 .empty .empty) .empty))) (.node 9337 (.node 7489 (.node 7309 (.node 6991 .empty .empty) (.node 7477 .empty .empty)) (.node 9283 (.node 7927 .empty .empty) .empty)) (.node 11953 (.node 11239 (.node 10459 .empty .empty) .empty) (.node 12697 (.node 12487 .empty .empty) .empty)))) (.node 19753 (.node 15541 (.node 14071 (.node 13681 (.node 13033 .empty .empty) (.node 13873 .empty .empty)) (.node 15361 (.node 14797 .empty .empty) .empty)) (.node 16981 (.node 16759 (.node 15607 .empty .empty) .empty) (.node 18661 (.node 18439 .empty .empty) .empty))) (.node 22051 (.node 20563 (.node 20341 (.node 20269 .empty .empty) (.node 20521 .empty .empty)) (.node 21601 (.node 20707 .empty .empty) .empty)) (.node 23227 (.node 22921 (.node 22147 .empty .empty) .empty) (.node 24121 (.node 23899 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1693_ordered : SortedNatTree.ordered row1693 = true := by decide
set_option maxRecDepth 4096 in
theorem row1693_values : SortedNatTree.values row1693 = Data.row1693 := rfl
theorem row1693_checked : SortedNatTree.contains row1693 = Data.row1693.contains := by
  rw [SortedNatTree.contains_eq_list row1693 row1693_ordered, row1693_values]
def row1697 : SortedNatTree.Tree := (.node 11681 (.node 7193 (.node 4931 (.node 2741 (.node 2027 (.node 1979 .empty .empty) .empty) (.node 3527 (.node 3041 .empty .empty) .empty)) (.node 5477 (.node 5099 (.node 5087 .empty .empty) .empty) (.node 6689 .empty .empty))) (.node 9467 (.node 8237 (.node 7901 (.node 7643 .empty .empty) .empty) (.node 9419 .empty .empty)) (.node 10391 (.node 10253 (.node 9887 .empty .empty) .empty) (.node 11483 .empty .empty)))) (.node 16901 (.node 13421 (.node 12527 (.node 12113 (.node 11813 .empty .empty) .empty) (.node 13151 (.node 12953 .empty .empty) .empty)) (.node 15629 (.node 14879 (.node 13697 .empty .empty) .empty) (.node 16061 .empty .empty))) (.node 21011 (.node 18797 (.node 18713 (.node 18617 .empty .empty) .empty) (.node 19661 .empty .empty)) (.node 22901 (.node 22109 (.node 21401 .empty .empty) .empty) (.node 23327 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1697_ordered : SortedNatTree.ordered row1697 = true := by decide
set_option maxRecDepth 4096 in
theorem row1697_values : SortedNatTree.values row1697 = Data.row1697 := rfl
theorem row1697_checked : SortedNatTree.contains row1697 = Data.row1697.contains := by
  rw [SortedNatTree.contains_eq_list row1697 row1697_ordered, row1697_values]
def row1699 : SortedNatTree.Tree := (.node 11317 (.node 4567 (.node 3313 (.node 2179 (.node 1861 (.node 1759 .empty .empty) .empty) (.node 2851 (.node 2203 .empty .empty) .empty)) (.node 3853 (.node 3607 (.node 3373 .empty .empty) .empty) (.node 4363 (.node 4177 .empty .empty) .empty))) (.node 7507 (.node 6121 (.node 5791 (.node 5659 .empty .empty) .empty) (.node 6691 (.node 6571 .empty .empty) .empty)) (.node 8713 (.node 7639 (.node 7549 .empty .empty) .empty) (.node 10957 (.node 10789 .empty .empty) .empty)))) (.node 15511 (.node 13033 (.node 12517 (.node 12343 (.node 11353 .empty .empty) .empty) (.node 12577 (.node 12553 .empty .empty) .empty)) (.node 14737 (.node 13399 (.node 13099 .empty .empty) .empty) (.node 15121 (.node 14923 .empty .empty) .empty))) (.node 21481 (.node 17569 (.node 17383 (.node 16339 .empty .empty) .empty) (.node 19609 (.node 18481 .empty .empty) .empty)) (.node 22783 (.node 21799 (.node 21727 .empty .empty) .empty) (.node 23497 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1699_ordered : SortedNatTree.ordered row1699 = true := by decide
set_option maxRecDepth 4096 in
theorem row1699_values : SortedNatTree.values row1699 = Data.row1699 := rfl
theorem row1699_checked : SortedNatTree.contains row1699 = Data.row1699.contains := by
  rw [SortedNatTree.contains_eq_list row1699 row1699_ordered, row1699_values]
def row1709 : SortedNatTree.Tree := (.node 9431 (.node 4397 (.node 3137 (.node 2393 (.node 2213 (.node 1913 .empty .empty) (.node 2333 .empty .empty)) (.node 2591 (.node 2423 .empty .empty) (.node 2843 .empty .empty))) (.node 4001 (.node 3389 (.node 3299 .empty .empty) (.node 3917 .empty .empty)) (.node 4283 (.node 4013 .empty .empty) .empty))) (.node 7823 (.node 6869 (.node 6551 (.node 4523 .empty .empty) (.node 6857 .empty .empty)) (.node 7247 (.node 7001 .empty .empty) .empty)) (.node 8819 (.node 8231 (.node 7877 .empty .empty) (.node 8297 .empty .empty)) (.node 9209 (.node 8831 .empty .empty) .empty)))) (.node 15971 (.node 12671 (.node 10139 (.node 9521 (.node 9491 .empty .empty) (.node 9923 .empty .empty)) (.node 12281 (.node 11003 .empty .empty) .empty)) (.node 14699 (.node 14543 (.node 13913 .empty .empty) (.node 14591 .empty .empty)) (.node 15137 (.node 15131 .empty .empty) .empty))) (.node 21977 (.node 19949 (.node 19379 (.node 19181 .empty .empty) (.node 19559 .empty .empty)) (.node 21323 (.node 20117 .empty .empty) .empty)) (.node 22751 (.node 22133 (.node 22091 .empty .empty) (.node 22541 .empty .empty)) (.node 24107 (.node 23201 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1709_ordered : SortedNatTree.ordered row1709 = true := by decide
set_option maxRecDepth 4096 in
theorem row1709_values : SortedNatTree.values row1709 = Data.row1709 := rfl
theorem row1709_checked : SortedNatTree.contains row1709 = Data.row1709.contains := by
  rw [SortedNatTree.contains_eq_list row1709 row1709_ordered, row1709_values]
def row1721 : SortedNatTree.Tree := (.node 10937 (.node 7829 (.node 3557 (.node 2621 (.node 2297 (.node 2213 .empty .empty) (.node 2411 .empty .empty)) (.node 3359 (.node 3347 .empty .empty) .empty)) (.node 6779 (.node 6389 (.node 4679 .empty .empty) .empty) (.node 7589 (.node 7283 .empty .empty) .empty))) (.node 8429 (.node 8231 (.node 8219 (.node 8039 .empty .empty) .empty) (.node 8387 (.node 8273 .empty .empty) .empty)) (.node 9677 (.node 8627 (.node 8573 .empty .empty) .empty) (.node 10079 (.node 9749 .empty .empty) .empty)))) (.node 15887 (.node 14771 (.node 14561 (.node 12107 (.node 11579 .empty .empty) .empty) (.node 14753 (.node 14717 .empty .empty) .empty)) (.node 15137 (.node 15107 (.node 14951 .empty .empty) .empty) (.node 15287 (.node 15149 .empty .empty) .empty))) (.node 19463 (.node 16871 (.node 16349 (.node 16223 .empty .empty) .empty) (.node 18341 (.node 18329 .empty .empty) .empty)) (.node 21557 (.node 20549 (.node 20177 .empty .empty) .empty) (.node 22697 (.node 21767 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1721_ordered : SortedNatTree.ordered row1721 = true := by decide
set_option maxRecDepth 4096 in
theorem row1721_values : SortedNatTree.values row1721 = Data.row1721 := rfl
theorem row1721_checked : SortedNatTree.contains row1721 = Data.row1721.contains := by
  rw [SortedNatTree.contains_eq_list row1721 row1721_ordered, row1721_values]
def row1723 : SortedNatTree.Tree := (.node 9337 (.node 5431 (.node 4423 (.node 3457 (.node 2179 (.node 1831 .empty .empty) (.node 3319 .empty .empty)) (.node 4363 (.node 4057 .empty .empty) .empty)) (.node 4993 (.node 4783 (.node 4621 .empty .empty) (.node 4813 .empty .empty)) (.node 5419 (.node 4999 .empty .empty) .empty))) (.node 8467 (.node 7993 (.node 6547 (.node 6397 .empty .empty) (.node 7951 .empty .empty)) (.node 8329 (.node 8233 .empty .empty) .empty)) (.node 8863 (.node 8623 (.node 8563 .empty .empty) .empty) (.node 9109 (.node 9067 .empty .empty) .empty)))) (.node 16699 (.node 12433 (.node 10711 (.node 10111 (.node 9643 .empty .empty) (.node 10627 .empty .empty)) (.node 12211 (.node 11467 .empty .empty) .empty)) (.node 15271 (.node 14767 (.node 13591 .empty .empty) .empty) (.node 15607 (.node 15289 .empty .empty) .empty))) (.node 20173 (.node 18301 (.node 16843 (.node 16741 .empty .empty) (.node 17377 .empty .empty)) (.node 19501 (.node 18637 .empty .empty) .empty)) (.node 22441 (.node 21997 (.node 21991 .empty .empty) .empty) (.node 24049 (.node 22483 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1723_ordered : SortedNatTree.ordered row1723 = true := by decide
set_option maxRecDepth 4096 in
theorem row1723_values : SortedNatTree.values row1723 = Data.row1723 := rfl
theorem row1723_checked : SortedNatTree.contains row1723 = Data.row1723.contains := by
  rw [SortedNatTree.contains_eq_list row1723 row1723_ordered, row1723_values]
def row1733 : SortedNatTree.Tree := (.node 8969 (.node 5273 (.node 3593 (.node 2663 (.node 2657 (.node 2411 .empty .empty) .empty) (.node 3347 (.node 2927 .empty .empty) .empty)) (.node 4133 (.node 4001 (.node 3917 .empty .empty) .empty) (.node 5051 (.node 4817 .empty .empty) .empty))) (.node 6959 (.node 6101 (.node 5897 (.node 5867 .empty .empty) .empty) (.node 6521 (.node 6323 .empty .empty) .empty)) (.node 7949 (.node 7757 (.node 7043 .empty .empty) .empty) (.node 8807 (.node 8627 .empty .empty) .empty)))) (.node 17393 (.node 12203 (.node 10337 (.node 9929 (.node 9719 .empty .empty) .empty) (.node 11273 (.node 10463 .empty .empty) .empty)) (.node 14891 (.node 13457 (.node 12401 .empty .empty) .empty) (.node 16607 (.node 16427 .empty .empty) .empty))) (.node 21149 (.node 18911 (.node 18503 (.node 18329 .empty .empty) .empty) (.node 20183 (.node 19379 .empty .empty) .empty)) (.node 23327 (.node 22901 (.node 21611 .empty .empty) .empty) (.node 24083 (.node 23531 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1733_ordered : SortedNatTree.ordered row1733 = true := by decide
set_option maxRecDepth 4096 in
theorem row1733_values : SortedNatTree.values row1733 = Data.row1733 := rfl
theorem row1733_checked : SortedNatTree.contains row1733 = Data.row1733.contains := by
  rw [SortedNatTree.contains_eq_list row1733 row1733_ordered, row1733_values]
def row1741 : SortedNatTree.Tree := (.node 8419 (.node 3607 (.node 2851 (.node 2281 (.node 2053 (.node 1753 .empty .empty) (.node 2203 .empty .empty)) (.node 2347 (.node 2287 .empty .empty) .empty)) (.node 3307 (.node 2953 (.node 2887 .empty .empty) (.node 2971 .empty .empty)) (.node 3373 (.node 3313 .empty .empty) .empty))) (.node 5581 (.node 5227 (.node 3919 (.node 3793 .empty .empty) (.node 4003 .empty .empty)) (.node 5557 (.node 5479 .empty .empty) .empty)) (.node 7573 (.node 7489 (.node 6397 .empty .empty) .empty) (.node 8287 (.node 7993 .empty .empty) .empty)))) (.node 15919 (.node 11887 (.node 9091 (.node 8893 (.node 8539 .empty .empty) (.node 9007 .empty .empty)) (.node 10837 (.node 10159 .empty .empty) .empty)) (.node 14029 (.node 13309 (.node 12973 .empty .empty) .empty) (.node 14887 (.node 14653 .empty .empty) .empty))) (.node 21169 (.node 19927 (.node 17737 (.node 17467 .empty .empty) (.node 18121 .empty .empty)) (.node 20641 (.node 20593 .empty .empty) .empty)) (.node 24061 (.node 23857 (.node 23623 .empty .empty) .empty) (.node 24169 (.node 24103 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1741_ordered : SortedNatTree.ordered row1741 = true := by decide
set_option maxRecDepth 4096 in
theorem row1741_values : SortedNatTree.values row1741 = Data.row1741 := rfl
theorem row1741_checked : SortedNatTree.contains row1741 = Data.row1741.contains := by
  rw [SortedNatTree.contains_eq_list row1741 row1741_ordered, row1741_values]
def row1747 : SortedNatTree.Tree := (.node 9403 (.node 6037 (.node 3769 (.node 2179 (.node 1933 (.node 1873 .empty .empty) (.node 2053 .empty .empty)) (.node 3691 (.node 2671 .empty .empty) .empty)) (.node 4993 (.node 4099 (.node 3889 .empty .empty) (.node 4231 .empty .empty)) (.node 5737 (.node 5569 .empty .empty) .empty))) (.node 7639 (.node 7129 (.node 6421 (.node 6367 .empty .empty) (.node 6967 .empty .empty)) (.node 7477 (.node 7177 .empty .empty) .empty)) (.node 8731 (.node 8563 (.node 8011 .empty .empty) (.node 8599 .empty .empty)) (.node 9157 (.node 9151 .empty .empty) .empty)))) (.node 17497 (.node 13093 (.node 12457 (.node 9811 (.node 9643 .empty .empty) (.node 12007 .empty .empty)) (.node 12973 (.node 12763 .empty .empty) .empty)) (.node 15643 (.node 13291 (.node 13249 .empty .empty) (.node 14347 .empty .empty)) (.node 16573 (.node 16453 .empty .empty) .empty))) (.node 20269 (.node 18169 (.node 17827 (.node 17707 .empty .empty) (.node 18121 .empty .empty)) (.node 19603 (.node 18211 .empty .empty) .empty)) (.node 23623 (.node 22501 (.node 20323 .empty .empty) .empty) (.node 24061 (.node 24049 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1747_ordered : SortedNatTree.ordered row1747 = true := by decide
set_option maxRecDepth 4096 in
theorem row1747_values : SortedNatTree.values row1747 = Data.row1747 := rfl
theorem row1747_checked : SortedNatTree.contains row1747 = Data.row1747.contains := by
  rw [SortedNatTree.contains_eq_list row1747 row1747_ordered, row1747_values]
def row1753 : SortedNatTree.Tree := (.node 8929 (.node 6199 (.node 4567 (.node 3643 (.node 1951 (.node 1873 .empty .empty) (.node 3547 .empty .empty)) (.node 3709 (.node 3673 .empty .empty) (.node 3847 .empty .empty))) (.node 5281 (.node 4987 (.node 4903 .empty .empty) (.node 5197 .empty .empty)) (.node 5851 (.node 5827 .empty .empty) .empty))) (.node 7549 (.node 7237 (.node 6823 (.node 6691 .empty .empty) (.node 6907 .empty .empty)) (.node 7411 (.node 7321 .empty .empty) .empty)) (.node 8161 (.node 7789 (.node 7669 .empty .empty) (.node 8089 .empty .empty)) (.node 8893 (.node 8887 .empty .empty) .empty)))) (.node 14389 (.node 11113 (.node 10159 (.node 9091 (.node 9007 .empty .empty) (.node 9433 .empty .empty)) (.node 10789 (.node 10657 .empty .empty) .empty)) (.node 13267 (.node 11839 (.node 11527 .empty .empty) (.node 12457 .empty .empty)) (.node 14107 (.node 13309 .empty .empty) .empty))) (.node 20947 (.node 17881 (.node 17341 (.node 15277 .empty .empty) (.node 17491 .empty .empty)) (.node 19717 (.node 18133 .empty .empty) .empty)) (.node 21859 (.node 21313 (.node 21121 .empty .empty) (.node 21499 .empty .empty)) (.node 23761 (.node 23209 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1753_ordered : SortedNatTree.ordered row1753 = true := by decide
set_option maxRecDepth 4096 in
theorem row1753_values : SortedNatTree.values row1753 = Data.row1753 := rfl
theorem row1753_checked : SortedNatTree.contains row1753 = Data.row1753.contains := by
  rw [SortedNatTree.contains_eq_list row1753 row1753_ordered, row1753_values]
def row1759 : SortedNatTree.Tree := (.node 9277 (.node 5689 (.node 3733 (.node 3001 (.node 2749 (.node 1831 .empty .empty) .empty) (.node 3571 (.node 3319 .empty .empty) .empty)) (.node 4177 (.node 3889 (.node 3847 .empty .empty) .empty) (.node 5233 .empty .empty))) (.node 6703 (.node 6367 (.node 6361 (.node 6199 .empty .empty) .empty) (.node 6637 .empty .empty)) (.node 8839 (.node 7417 (.node 7351 .empty .empty) .empty) (.node 8941 .empty .empty)))) (.node 16141 (.node 13567 (.node 10687 (.node 9973 (.node 9613 .empty .empty) .empty) (.node 12853 (.node 12049 .empty .empty) .empty)) (.node 15307 (.node 15121 (.node 13879 .empty .empty) .empty) (.node 15331 .empty .empty))) (.node 18493 (.node 17239 (.node 16729 (.node 16693 .empty .empty) .empty) (.node 18451 .empty .empty)) (.node 22063 (.node 19441 (.node 18661 .empty .empty) .empty) (.node 23581 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1759_ordered : SortedNatTree.ordered row1759 = true := by decide
set_option maxRecDepth 4096 in
theorem row1759_values : SortedNatTree.values row1759 = Data.row1759 := rfl
theorem row1759_checked : SortedNatTree.contains row1759 = Data.row1759.contains := by
  rw [SortedNatTree.contains_eq_list row1759 row1759_ordered, row1759_values]
def row1777 : SortedNatTree.Tree := (.node 10597 (.node 5839 (.node 3121 (.node 2539 (.node 2371 (.node 2017 .empty .empty) .empty) (.node 2719 (.node 2647 .empty .empty) .empty)) (.node 3739 (.node 3463 (.node 3253 .empty .empty) .empty) (.node 5011 (.node 4621 .empty .empty) .empty))) (.node 7603 (.node 7213 (.node 6277 (.node 6151 .empty .empty) .empty) (.node 7561 (.node 7489 .empty .empty) .empty)) (.node 9439 (.node 7867 (.node 7741 .empty .empty) .empty) (.node 10177 (.node 9787 .empty .empty) .empty)))) (.node 16183 (.node 12979 (.node 12433 (.node 12211 (.node 11059 .empty .empty) .empty) (.node 12823 (.node 12487 .empty .empty) .empty)) (.node 15121 (.node 14683 (.node 13693 .empty .empty) .empty) (.node 15991 (.node 15271 .empty .empty) .empty))) (.node 21067 (.node 19891 (.node 19699 (.node 18523 .empty .empty) .empty) (.node 20353 (.node 20143 .empty .empty) .empty)) (.node 23563 (.node 23209 (.node 21601 .empty .empty) .empty) (.node 24109 (.node 23671 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1777_ordered : SortedNatTree.ordered row1777 = true := by decide
set_option maxRecDepth 4096 in
theorem row1777_values : SortedNatTree.values row1777 = Data.row1777 := rfl
theorem row1777_checked : SortedNatTree.contains row1777 = Data.row1777.contains := by
  rw [SortedNatTree.contains_eq_list row1777 row1777_ordered, row1777_values]
def row1783 : SortedNatTree.Tree := (.node 12109 (.node 6373 (.node 4273 (.node 2953 (.node 2707 (.node 2053 .empty .empty) (.node 2791 .empty .empty)) (.node 3769 (.node 3643 .empty .empty) .empty)) (.node 6121 (.node 5011 (.node 4639 .empty .empty) .empty) (.node 6277 (.node 6211 .empty .empty) .empty))) (.node 9067 (.node 8209 (.node 7129 (.node 6823 .empty .empty) .empty) (.node 8647 (.node 8461 .empty .empty) .empty)) (.node 11587 (.node 11083 (.node 9721 .empty .empty) .empty) (.node 11719 (.node 11689 .empty .empty) .empty)))) (.node 17443 (.node 14947 (.node 14341 (.node 14293 (.node 12391 .empty .empty) .empty) (.node 14713 (.node 14551 .empty .empty) .empty)) (.node 15727 (.node 15583 (.node 15121 .empty .empty) .empty) (.node 16903 (.node 15919 .empty .empty) .empty))) (.node 21937 (.node 20161 (.node 19489 (.node 18367 .empty .empty) .empty) (.node 21391 (.node 20551 .empty .empty) .empty)) (.node 23773 (.node 23767 (.node 22003 .empty .empty) .empty) (.node 24151 (.node 24061 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1783_ordered : SortedNatTree.ordered row1783 = true := by decide
set_option maxRecDepth 4096 in
theorem row1783_values : SortedNatTree.values row1783 = Data.row1783 := rfl
theorem row1783_checked : SortedNatTree.contains row1783 = Data.row1783.contains := by
  rw [SortedNatTree.contains_eq_list row1783 row1783_ordered, row1783_values]
def row1787 : SortedNatTree.Tree := (.node 9791 (.node 6269 (.node 4919 (.node 4271 (.node 3761 (.node 2447 .empty .empty) (.node 4211 .empty .empty)) (.node 4391 (.node 4349 .empty .empty) .empty)) (.node 5591 (.node 5387 (.node 4937 .empty .empty) .empty) (.node 6011 (.node 5927 .empty .empty) .empty))) (.node 7901 (.node 7547 (.node 6983 (.node 6323 .empty .empty) .empty) (.node 7823 (.node 7757 .empty .empty) .empty)) (.node 8681 (.node 8447 (.node 8387 .empty .empty) .empty) (.node 9479 (.node 9221 .empty .empty) .empty)))) (.node 17471 (.node 13001 (.node 11801 (.node 10733 (.node 10457 .empty .empty) (.node 11003 .empty .empty)) (.node 12269 (.node 11933 .empty .empty) .empty)) (.node 14621 (.node 14243 (.node 13553 .empty .empty) .empty) (.node 15959 (.node 15671 .empty .empty) .empty))) (.node 19661 (.node 18443 (.node 18401 (.node 17921 .empty .empty) .empty) (.node 18797 (.node 18719 .empty .empty) .empty)) (.node 22871 (.node 22541 (.node 20753 .empty .empty) .empty) (.node 23813 (.node 22937 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1787_ordered : SortedNatTree.ordered row1787 = true := by decide
set_option maxRecDepth 4096 in
theorem row1787_values : SortedNatTree.values row1787 = Data.row1787 := rfl
theorem row1787_checked : SortedNatTree.contains row1787 = Data.row1787.contains := by
  rw [SortedNatTree.contains_eq_list row1787 row1787_ordered, row1787_values]
def row1789 : SortedNatTree.Tree := (.node 9907 (.node 5023 (.node 2887 (.node 2503 (.node 2377 (.node 2089 .empty .empty) (.node 2467 .empty .empty)) (.node 2683 (.node 2521 .empty .empty) .empty)) (.node 4243 (.node 3907 (.node 3853 .empty .empty) (.node 4111 .empty .empty)) (.node 4831 (.node 4813 .empty .empty) .empty))) (.node 7393 (.node 6421 (.node 6133 (.node 5953 .empty .empty) (.node 6343 .empty .empty)) (.node 6661 (.node 6547 .empty .empty) .empty)) (.node 9439 (.node 8641 (.node 8011 .empty .empty) .empty) (.node 9811 (.node 9619 .empty .empty) .empty)))) (.node 15277 (.node 12583 (.node 12157 (.node 11197 (.node 10789 .empty .empty) (.node 11497 .empty .empty)) (.node 12241 (.node 12211 .empty .empty) .empty)) (.node 13003 (.node 12763 (.node 12703 .empty .empty) .empty) (.node 14479 (.node 14389 .empty .empty) .empty))) (.node 19381 (.node 17971 (.node 16069 (.node 15391 .empty .empty) (.node 17449 .empty .empty)) (.node 18919 (.node 18757 .empty .empty) .empty)) (.node 23131 (.node 20857 (.node 20479 .empty .empty) .empty) (.node 23761 (.node 23539 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1789_ordered : SortedNatTree.ordered row1789 = true := by decide
set_option maxRecDepth 4096 in
theorem row1789_values : SortedNatTree.values row1789 = Data.row1789 := rfl
theorem row1789_checked : SortedNatTree.contains row1789 = Data.row1789.contains := by
  rw [SortedNatTree.contains_eq_list row1789 row1789_ordered, row1789_values]
def row1801 : SortedNatTree.Tree := (.node 8233 (.node 5167 (.node 3613 (.node 3253 (.node 2647 (.node 2311 .empty .empty) .empty) (.node 3547 .empty .empty)) (.node 4177 (.node 3823 .empty .empty) (.node 4933 .empty .empty))) (.node 6907 (.node 6637 (.node 5689 .empty .empty) (.node 6829 .empty .empty)) (.node 7621 (.node 7333 .empty .empty) (.node 7669 .empty .empty)))) (.node 18691 (.node 13537 (.node 11353 (.node 11149 .empty .empty) (.node 12277 .empty .empty)) (.node 14821 (.node 14593 .empty .empty) (.node 18181 .empty .empty))) (.node 21589 (.node 20533 (.node 18919 .empty .empty) (.node 21397 .empty .empty)) (.node 23227 (.node 22303 .empty .empty) (.node 23539 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1801_ordered : SortedNatTree.ordered row1801 = true := by decide
set_option maxRecDepth 4096 in
theorem row1801_values : SortedNatTree.values row1801 = Data.row1801 := rfl
theorem row1801_checked : SortedNatTree.contains row1801 = Data.row1801.contains := by
  rw [SortedNatTree.contains_eq_list row1801 row1801_ordered, row1801_values]
def row1811 : SortedNatTree.Tree := (.node 13901 (.node 6863 (.node 3257 (.node 2333 (.node 1949 (.node 1931 .empty .empty) (.node 2141 .empty .empty)) (.node 2609 (.node 2441 .empty .empty) .empty)) (.node 5381 (.node 5297 (.node 5039 .empty .empty) .empty) (.node 6653 (.node 6473 .empty .empty) .empty))) (.node 10733 (.node 8699 (.node 7817 (.node 7559 .empty .empty) (.node 8669 .empty .empty)) (.node 10691 (.node 9281 .empty .empty) .empty)) (.node 12011 (.node 11423 (.node 10799 .empty .empty) .empty) (.node 13859 (.node 12263 .empty .empty) .empty)))) (.node 18713 (.node 16451 (.node 14699 (.node 14033 (.node 14009 .empty .empty) (.node 14159 .empty .empty)) (.node 15881 (.node 15287 .empty .empty) .empty)) (.node 16631 (.node 16619 (.node 16547 .empty .empty) .empty) (.node 17477 (.node 17207 .empty .empty) .empty))) (.node 23057 (.node 21821 (.node 21491 (.node 19889 .empty .empty) .empty) (.node 22937 (.node 21911 .empty .empty) .empty)) (.node 24071 (.node 23417 (.node 23081 .empty .empty) .empty) (.node 24113 (.node 24107 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1811_ordered : SortedNatTree.ordered row1811 = true := by decide
set_option maxRecDepth 4096 in
theorem row1811_values : SortedNatTree.values row1811 = Data.row1811 := rfl
theorem row1811_checked : SortedNatTree.contains row1811 = Data.row1811.contains := by
  rw [SortedNatTree.contains_eq_list row1811 row1811_ordered, row1811_values]
def row1823 : SortedNatTree.Tree := (.node 11903 (.node 6353 (.node 3461 (.node 2111 (.node 2069 (.node 2063 .empty .empty) .empty) (.node 3023 (.node 2861 .empty .empty) .empty)) (.node 4493 (.node 4001 (.node 3947 .empty .empty) .empty) (.node 6131 .empty .empty))) (.node 8783 (.node 6983 (.node 6947 (.node 6719 .empty .empty) .empty) (.node 8069 (.node 7823 .empty .empty) .empty)) (.node 10853 (.node 10631 (.node 9341 .empty .empty) .empty) (.node 10859 .empty .empty)))) (.node 16097 (.node 14159 (.node 12203 (.node 12161 (.node 12149 .empty .empty) .empty) (.node 13997 (.node 12659 .empty .empty) .empty)) (.node 14843 (.node 14669 (.node 14423 .empty .empty) .empty) (.node 15629 .empty .empty))) (.node 19469 (.node 18671 (.node 17957 (.node 17189 .empty .empty) .empty) (.node 19001 .empty .empty)) (.node 23993 (.node 23117 (.node 21767 .empty .empty) .empty) (.node 24029 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1823_ordered : SortedNatTree.ordered row1823 = true := by decide
set_option maxRecDepth 4096 in
theorem row1823_values : SortedNatTree.values row1823 = Data.row1823 := rfl
theorem row1823_checked : SortedNatTree.contains row1823 = Data.row1823.contains := by
  rw [SortedNatTree.contains_eq_list row1823 row1823_ordered, row1823_values]
def row1831 : SortedNatTree.Tree := (.node 13003 (.node 9241 (.node 6301 (.node 3769 (.node 3259 (.node 2803 .empty .empty) .empty) (.node 5197 .empty .empty)) (.node 8233 (.node 7927 (.node 7603 .empty .empty) .empty) (.node 8941 .empty .empty))) (.node 11239 (.node 10729 (.node 10663 (.node 10651 .empty .empty) .empty) (.node 10903 .empty .empty)) (.node 11491 (.node 11287 .empty .empty) (.node 12421 .empty .empty)))) (.node 16651 (.node 15073 (.node 14143 (.node 13729 (.node 13309 .empty .empty) .empty) (.node 14389 .empty .empty)) (.node 15937 (.node 15817 .empty .empty) (.node 16447 .empty .empty))) (.node 21319 (.node 20149 (.node 20029 (.node 17707 .empty .empty) .empty) (.node 20611 .empty .empty)) (.node 22447 (.node 21613 .empty .empty) (.node 22531 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1831_ordered : SortedNatTree.ordered row1831 = true := by decide
set_option maxRecDepth 4096 in
theorem row1831_values : SortedNatTree.values row1831 = Data.row1831 := rfl
theorem row1831_checked : SortedNatTree.contains row1831 = Data.row1831.contains := by
  rw [SortedNatTree.contains_eq_list row1831 row1831_ordered, row1831_values]
def row1847 : SortedNatTree.Tree := (.node 8951 (.node 4919 (.node 3863 (.node 3011 (.node 2687 (.node 2423 .empty .empty) (.node 2969 .empty .empty)) (.node 3779 (.node 3671 .empty .empty) .empty)) (.node 4373 (.node 4283 (.node 4007 .empty .empty) .empty) (.node 4691 (.node 4457 .empty .empty) .empty))) (.node 7523 (.node 5927 (.node 5801 (.node 5519 .empty .empty) .empty) (.node 7307 (.node 6173 .empty .empty) .empty)) (.node 8123 (.node 8069 (.node 7703 .empty .empty) .empty) (.node 8783 (.node 8693 .empty .empty) .empty)))) (.node 17657 (.node 12071 (.node 9941 (.node 9743 (.node 9311 .empty .empty) .empty) (.node 10613 (.node 10259 .empty .empty) .empty)) (.node 15299 (.node 13913 (.node 13229 .empty .empty) .empty) (.node 16139 (.node 15767 .empty .empty) .empty))) (.node 20051 (.node 17837 (.node 17783 (.node 17681 .empty .empty) .empty) (.node 19991 (.node 18257 .empty .empty) .empty)) (.node 23447 (.node 21599 (.node 21521 .empty .empty) .empty) (.node 24107 (.node 23663 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1847_ordered : SortedNatTree.ordered row1847 = true := by decide
set_option maxRecDepth 4096 in
theorem row1847_values : SortedNatTree.values row1847 = Data.row1847 := rfl
theorem row1847_checked : SortedNatTree.contains row1847 = Data.row1847.contains := by
  rw [SortedNatTree.contains_eq_list row1847 row1847_ordered, row1847_values]
def row1861 : SortedNatTree.Tree := (.node 7867 (.node 5011 (.node 3163 (.node 2371 (.node 1951 (.node 1933 .empty .empty) .empty) (.node 2791 (.node 2383 .empty .empty) .empty)) (.node 3769 (.node 3709 (.node 3433 .empty .empty) .empty) (.node 4789 .empty .empty))) (.node 6007 (.node 5521 (.node 5479 (.node 5059 .empty .empty) .empty) (.node 5881 .empty .empty)) (.node 6907 (.node 6133 (.node 6121 .empty .empty) .empty) (.node 7213 .empty .empty)))) (.node 15271 (.node 12211 (.node 10627 (.node 10501 (.node 9973 .empty .empty) .empty) (.node 11551 (.node 11437 .empty .empty) .empty)) (.node 14983 (.node 14869 (.node 13591 .empty .empty) .empty) (.node 15187 .empty .empty))) (.node 18043 (.node 17137 (.node 16747 (.node 16477 .empty .empty) .empty) (.node 17599 .empty .empty)) (.node 22291 (.node 22159 (.node 19993 .empty .empty) .empty) (.node 22567 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1861_ordered : SortedNatTree.ordered row1861 = true := by decide
set_option maxRecDepth 4096 in
theorem row1861_values : SortedNatTree.values row1861 = Data.row1861 := rfl
theorem row1861_checked : SortedNatTree.contains row1861 = Data.row1861.contains := by
  rw [SortedNatTree.contains_eq_list row1861 row1861_ordered, row1861_values]
def row1867 : SortedNatTree.Tree := (.node 12157 (.node 7873 (.node 5659 (.node 3673 (.node 2221 (.node 2053 .empty .empty) .empty) (.node 5281 (.node 4111 .empty .empty) .empty)) (.node 6733 (.node 6121 (.node 5749 .empty .empty) .empty) (.node 7639 (.node 7129 .empty .empty) .empty))) (.node 10831 (.node 8731 (.node 8713 (.node 8293 .empty .empty) .empty) (.node 10789 (.node 10753 .empty .empty) .empty)) (.node 11353 (.node 11299 (.node 11131 .empty .empty) .empty) (.node 11887 .empty .empty)))) (.node 18913 (.node 15241 (.node 14923 (.node 14029 (.node 13831 .empty .empty) .empty) (.node 15091 (.node 15031 .empty .empty) .empty)) (.node 18061 (.node 16693 (.node 16381 .empty .empty) .empty) (.node 18313 .empty .empty))) (.node 20407 (.node 19333 (.node 19237 (.node 19069 .empty .empty) .empty) (.node 20071 (.node 19447 .empty .empty) .empty)) (.node 23371 (.node 22447 (.node 21019 .empty .empty) .empty) (.node 24151 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1867_ordered : SortedNatTree.ordered row1867 = true := by decide
set_option maxRecDepth 4096 in
theorem row1867_values : SortedNatTree.values row1867 = Data.row1867 := rfl
theorem row1867_checked : SortedNatTree.contains row1867 = Data.row1867.contains := by
  rw [SortedNatTree.contains_eq_list row1867 row1867_ordered, row1867_values]
def row1871 : SortedNatTree.Tree := (.node 13127 (.node 7001 (.node 4733 (.node 4517 (.node 4139 (.node 3137 .empty .empty) (.node 4349 .empty .empty)) (.node 4691 (.node 4649 .empty .empty) .empty)) (.node 5441 (.node 5231 (.node 5051 .empty .empty) .empty) (.node 6959 (.node 6323 .empty .empty) .empty))) (.node 9479 (.node 8807 (.node 8681 (.node 8171 .empty .empty) .empty) (.node 9209 (.node 8969 .empty .empty) .empty)) (.node 11927 (.node 11789 (.node 9791 .empty .empty) .empty) (.node 12503 (.node 12119 .empty .empty) .empty)))) (.node 19661 (.node 17033 (.node 15881 (.node 15581 (.node 14747 .empty .empty) .empty) (.node 16931 (.node 16451 .empty .empty) .empty)) (.node 17921 (.node 17747 (.node 17729 .empty .empty) .empty) (.node 19013 (.node 18059 .empty .empty) .empty))) (.node 21377 (.node 19991 (.node 19949 (.node 19853 .empty .empty) .empty) (.node 21149 (.node 20333 .empty .empty) .empty)) (.node 22511 (.node 22157 (.node 22037 .empty .empty) .empty) (.node 22973 (.node 22853 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1871_ordered : SortedNatTree.ordered row1871 = true := by decide
set_option maxRecDepth 4096 in
theorem row1871_values : SortedNatTree.values row1871 = Data.row1871 := rfl
theorem row1871_checked : SortedNatTree.contains row1871 = Data.row1871.contains := by
  rw [SortedNatTree.contains_eq_list row1871 row1871_ordered, row1871_values]
def row1873 : SortedNatTree.Tree := (.node 9769 (.node 6961 (.node 5527 (.node 2437 (.node 2269 (.node 2017 .empty .empty) .empty) (.node 3511 (.node 2719 .empty .empty) .empty)) (.node 6427 (.node 6301 (.node 5569 .empty .empty) .empty) (.node 6841 (.node 6709 .empty .empty) .empty))) (.node 8179 (.node 7963 (.node 7507 (.node 6997 .empty .empty) .empty) (.node 8059 (.node 8011 .empty .empty) .empty)) (.node 9433 (.node 9241 (.node 9067 .empty .empty) .empty) (.node 9697 .empty .empty)))) (.node 14851 (.node 12967 (.node 10867 (.node 10657 (.node 9907 .empty .empty) .empty) (.node 12049 (.node 11251 .empty .empty) .empty)) (.node 14107 (.node 13627 (.node 13003 .empty .empty) .empty) (.node 14821 (.node 14437 .empty .empty) .empty))) (.node 21613 (.node 15493 (.node 15391 (.node 14947 .empty .empty) .empty) (.node 20143 (.node 15607 .empty .empty) .empty)) (.node 23167 (.node 23011 (.node 21943 .empty .empty) .empty) (.node 24091 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1873_ordered : SortedNatTree.ordered row1873 = true := by decide
set_option maxRecDepth 4096 in
theorem row1873_values : SortedNatTree.values row1873 = Data.row1873 := rfl
theorem row1873_checked : SortedNatTree.contains row1873 = Data.row1873.contains := by
  rw [SortedNatTree.contains_eq_list row1873 row1873_ordered, row1873_values]
def row1877 : SortedNatTree.Tree := (.node 12197 (.node 6287 (.node 4391 (.node 2909 (.node 2447 (.node 1889 .empty .empty) (.node 2549 .empty .empty)) (.node 3917 (.node 3119 .empty .empty) (.node 4211 .empty .empty))) (.node 5399 (.node 4733 (.node 4421 .empty .empty) (.node 5261 .empty .empty)) (.node 6113 (.node 5711 .empty .empty) .empty))) (.node 9209 (.node 7823 (.node 6971 (.node 6491 .empty .empty) (.node 6983 .empty .empty)) (.node 8597 (.node 8087 .empty .empty) .empty)) (.node 10739 (.node 9851 (.node 9533 .empty .empty) (.node 10499 .empty .empty)) (.node 11087 (.node 11003 .empty .empty) .empty)))) (.node 19751 (.node 14717 (.node 13931 (.node 13037 (.node 12809 .empty .empty) (.node 13499 .empty .empty)) (.node 14627 (.node 14591 .empty .empty) .empty)) (.node 17333 (.node 16223 (.node 15263 .empty .empty) (.node 16649 .empty .empty)) (.node 19211 (.node 18731 .empty .empty) .empty))) (.node 22751 (.node 21851 (.node 20441 (.node 20177 .empty .empty) (.node 21323 .empty .empty)) (.node 22397 (.node 22229 .empty .empty) .empty)) (.node 23561 (.node 23087 (.node 23057 .empty .empty) (.node 23201 .empty .empty)) (.node 24107 (.node 24077 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1877_ordered : SortedNatTree.ordered row1877 = true := by decide
set_option maxRecDepth 4096 in
theorem row1877_values : SortedNatTree.values row1877 = Data.row1877 := rfl
theorem row1877_checked : SortedNatTree.contains row1877 = Data.row1877.contains := by
  rw [SortedNatTree.contains_eq_list row1877 row1877_ordered, row1877_values]
def row1879 : SortedNatTree.Tree := (.node 12343 (.node 7411 (.node 2971 (.node 2341 (.node 2281 (.node 1999 .empty .empty) .empty) (.node 2713 .empty .empty)) (.node 6823 (.node 4861 (.node 3331 .empty .empty) .empty) (.node 7297 .empty .empty))) (.node 9883 (.node 8971 (.node 8707 (.node 7951 .empty .empty) .empty) (.node 9103 .empty .empty)) (.node 10267 (.node 9967 .empty .empty) (.node 12301 .empty .empty)))) (.node 16189 (.node 12763 (.node 12457 (.node 12409 (.node 12379 .empty .empty) .empty) (.node 12541 .empty .empty)) (.node 15307 (.node 13513 (.node 13183 .empty .empty) .empty) (.node 15619 .empty .empty))) (.node 19267 (.node 17791 (.node 17203 (.node 16903 .empty .empty) .empty) (.node 19183 .empty .empty)) (.node 22621 (.node 21817 .empty .empty) (.node 24043 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1879_ordered : SortedNatTree.ordered row1879 = true := by decide
set_option maxRecDepth 4096 in
theorem row1879_values : SortedNatTree.values row1879 = Data.row1879 := rfl
theorem row1879_checked : SortedNatTree.contains row1879 = Data.row1879.contains := by
  rw [SortedNatTree.contains_eq_list row1879 row1879_ordered, row1879_values]
def row1889 : SortedNatTree.Tree := (.node 10289 (.node 5417 (.node 2927 (.node 2633 (.node 2129 (.node 2087 .empty .empty) (.node 2243 .empty .empty)) (.node 2843 (.node 2777 .empty .empty) .empty)) (.node 4019 (.node 3491 (.node 2999 .empty .empty) .empty) (.node 5297 (.node 4919 .empty .empty) .empty))) (.node 9533 (.node 8123 (.node 7019 (.node 6833 .empty .empty) .empty) (.node 8861 (.node 8291 .empty .empty) .empty)) (.node 9941 (.node 9929 (.node 9539 .empty .empty) .empty) (.node 10193 (.node 10181 .empty .empty) .empty)))) (.node 15443 (.node 12107 (.node 10889 (.node 10529 (.node 10457 .empty .empty) (.node 10781 .empty .empty)) (.node 12011 (.node 11717 .empty .empty) .empty)) (.node 13499 (.node 13187 (.node 12527 .empty .empty) .empty) (.node 14753 (.node 13619 .empty .empty) .empty))) (.node 19697 (.node 17609 (.node 17291 (.node 15791 .empty .empty) .empty) (.node 18617 (.node 18077 .empty .empty) .empty)) (.node 23003 (.node 22901 (.node 22067 .empty .empty) .empty) (.node 23993 (.node 23561 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1889_ordered : SortedNatTree.ordered row1889 = true := by decide
set_option maxRecDepth 4096 in
theorem row1889_values : SortedNatTree.values row1889 = Data.row1889 := rfl
theorem row1889_checked : SortedNatTree.contains row1889 = Data.row1889.contains := by
  rw [SortedNatTree.contains_eq_list row1889 row1889_ordered, row1889_values]
def row1901 : SortedNatTree.Tree := (.node 9851 (.node 7589 (.node 3221 (.node 2801 (.node 2621 (.node 2153 .empty .empty) .empty) (.node 2969 .empty .empty)) (.node 5861 (.node 4817 (.node 4019 .empty .empty) .empty) (.node 7151 .empty .empty))) (.node 9227 (.node 8663 (.node 8243 (.node 7949 .empty .empty) .empty) (.node 8837 .empty .empty)) (.node 9467 (.node 9431 .empty .empty) (.node 9587 .empty .empty)))) (.node 17489 (.node 14957 (.node 14669 (.node 10889 (.node 10313 .empty .empty) .empty) (.node 14891 .empty .empty)) (.node 15923 (.node 15383 .empty .empty) (.node 15971 .empty .empty))) (.node 19913 (.node 18947 (.node 18371 (.node 18041 .empty .empty) .empty) (.node 19751 .empty .empty)) (.node 20939 (.node 20753 .empty .empty) (.node 22643 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1901_ordered : SortedNatTree.ordered row1901 = true := by decide
set_option maxRecDepth 4096 in
theorem row1901_values : SortedNatTree.values row1901 = Data.row1901 := rfl
theorem row1901_checked : SortedNatTree.contains row1901 = Data.row1901.contains := by
  rw [SortedNatTree.contains_eq_list row1901 row1901_ordered, row1901_values]
def row1907 : SortedNatTree.Tree := (.node 12227 (.node 5897 (.node 3581 (.node 2621 (.node 2243 (.node 2063 .empty .empty) .empty) (.node 3083 (.node 2777 .empty .empty) .empty)) (.node 4637 (.node 4127 (.node 3797 .empty .empty) .empty) (.node 5669 (.node 5099 .empty .empty) .empty))) (.node 9341 (.node 8819 (.node 8231 (.node 7481 .empty .empty) .empty) (.node 9041 (.node 8867 .empty .empty) .empty)) (.node 10337 (.node 10103 (.node 9677 .empty .empty) .empty) (.node 11867 (.node 11273 .empty .empty) .empty)))) (.node 21191 (.node 16301 (.node 14753 (.node 14543 (.node 14369 .empty .empty) .empty) (.node 16007 (.node 15731 .empty .empty) .empty)) (.node 17489 (.node 17327 (.node 16763 .empty .empty) .empty) (.node 20183 (.node 18251 .empty .empty) .empty))) (.node 22619 (.node 22229 (.node 21977 (.node 21767 .empty .empty) .empty) (.node 22613 (.node 22277 .empty .empty) .empty)) (.node 23831 (.node 23399 (.node 22907 .empty .empty) .empty) (.node 24077 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1907_ordered : SortedNatTree.ordered row1907 = true := by decide
set_option maxRecDepth 4096 in
theorem row1907_values : SortedNatTree.values row1907 = Data.row1907 := rfl
theorem row1907_checked : SortedNatTree.contains row1907 = Data.row1907.contains := by
  rw [SortedNatTree.contains_eq_list row1907 row1907_ordered, row1907_values]
def row1913 : SortedNatTree.Tree := (.node 9629 (.node 5483 (.node 3203 (.node 2693 (.node 2273 (.node 2213 .empty .empty) (.node 2381 .empty .empty)) (.node 2843 (.node 2741 .empty .empty) .empty)) (.node 5087 (.node 5009 (.node 4211 .empty .empty) .empty) (.node 5381 (.node 5189 .empty .empty) .empty))) (.node 7841 (.node 6389 (.node 6011 (.node 5861 .empty .empty) .empty) (.node 7583 (.node 7253 .empty .empty) .empty)) (.node 8297 (.node 8219 (.node 7919 .empty .empty) .empty) (.node 8807 (.node 8699 .empty .empty) .empty)))) (.node 16361 (.node 11789 (.node 10529 (.node 10169 (.node 9851 .empty .empty) (.node 10193 .empty .empty)) (.node 10949 (.node 10739 .empty .empty) .empty)) (.node 14723 (.node 13781 (.node 12119 .empty .empty) .empty) (.node 14831 (.node 14771 .empty .empty) .empty))) (.node 18839 (.node 17351 (.node 17123 (.node 16889 .empty .empty) .empty) (.node 18671 (.node 18593 .empty .empty) .empty)) (.node 22091 (.node 21617 (.node 21503 .empty .empty) .empty) (.node 23801 (.node 22871 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1913_ordered : SortedNatTree.ordered row1913 = true := by decide
set_option maxRecDepth 4096 in
theorem row1913_values : SortedNatTree.values row1913 = Data.row1913 := rfl
theorem row1913_checked : SortedNatTree.contains row1913 = Data.row1913.contains := by
  rw [SortedNatTree.contains_eq_list row1913 row1913_ordered, row1913_values]
def row1931 : SortedNatTree.Tree := (.node 13103 (.node 9437 (.node 5279 (.node 4289 (.node 2957 (.node 2399 .empty .empty) .empty) (.node 5171 (.node 4943 .empty .empty) .empty)) (.node 6719 (.node 5477 (.node 5297 .empty .empty) .empty) (.node 8753 (.node 7043 .empty .empty) .empty))) (.node 10457 (.node 10007 (.node 9929 (.node 9533 .empty .empty) .empty) (.node 10343 (.node 10133 .empty .empty) .empty)) (.node 11393 (.node 11213 (.node 11057 .empty .empty) .empty) (.node 11813 .empty .empty)))) (.node 17921 (.node 15287 (.node 14057 (.node 13331 (.node 13163 .empty .empty) .empty) (.node 14759 (.node 14447 .empty .empty) .empty)) (.node 16673 (.node 15551 (.node 15401 .empty .empty) .empty) (.node 17471 .empty .empty))) (.node 21893 (.node 19571 (.node 19037 (.node 18773 .empty .empty) .empty) (.node 19841 (.node 19583 .empty .empty) .empty)) (.node 23549 (.node 23531 (.node 22679 .empty .empty) .empty) (.node 23603 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1931_ordered : SortedNatTree.ordered row1931 = true := by decide
set_option maxRecDepth 4096 in
theorem row1931_values : SortedNatTree.values row1931 = Data.row1931 := rfl
theorem row1931_checked : SortedNatTree.contains row1931 = Data.row1931.contains := by
  rw [SortedNatTree.contains_eq_list row1931 row1931_ordered, row1931_values]
def row1933 : SortedNatTree.Tree := (.node 12799 (.node 7639 (.node 5023 (.node 3733 (.node 3313 (.node 2293 .empty .empty) .empty) (.node 5011 (.node 4201 .empty .empty) .empty)) (.node 6841 (.node 6703 (.node 5791 .empty .empty) .empty) (.node 7333 .empty .empty))) (.node 9391 (.node 8521 (.node 8287 (.node 7933 .empty .empty) .empty) (.node 8677 .empty .empty)) (.node 10627 (.node 10597 (.node 9907 .empty .empty) .empty) (.node 11383 .empty .empty)))) (.node 18859 (.node 17107 (.node 14533 (.node 13411 (.node 13297 .empty .empty) .empty) (.node 16369 (.node 15583 .empty .empty) .empty)) (.node 18481 (.node 17569 (.node 17401 .empty .empty) .empty) (.node 18523 .empty .empty))) (.node 21727 (.node 20593 (.node 19993 (.node 19267 .empty .empty) .empty) (.node 20887 .empty .empty)) (.node 23311 (.node 23197 (.node 23071 .empty .empty) .empty) (.node 23857 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1933_ordered : SortedNatTree.ordered row1933 = true := by decide
set_option maxRecDepth 4096 in
theorem row1933_values : SortedNatTree.values row1933 = Data.row1933 := rfl
theorem row1933_checked : SortedNatTree.contains row1933 = Data.row1933.contains := by
  rw [SortedNatTree.contains_eq_list row1933 row1933_ordered, row1933_values]
def row1949 : SortedNatTree.Tree := (.node 10889 (.node 4337 (.node 2819 (.node 2657 (.node 2621 (.node 2027 .empty .empty) .empty) (.node 2789 (.node 2663 .empty .empty) .empty)) (.node 3923 (.node 3911 (.node 2903 .empty .empty) .empty) (.node 3989 .empty .empty))) (.node 9749 (.node 7727 (.node 6863 (.node 5669 .empty .empty) .empty) (.node 9041 (.node 7901 .empty .empty) .empty)) (.node 10589 (.node 10259 (.node 10007 .empty .empty) .empty) (.node 10847 .empty .empty)))) (.node 19961 (.node 17159 (.node 14051 (.node 11939 (.node 11447 .empty .empty) .empty) (.node 15473 (.node 15263 .empty .empty) .empty)) (.node 18461 (.node 18233 (.node 17747 .empty .empty) .empty) (.node 19427 .empty .empty))) (.node 21773 (.node 20261 (.node 19997 (.node 19973 .empty .empty) .empty) (.node 21317 .empty .empty)) (.node 23789 (.node 23741 (.node 23369 .empty .empty) .empty) (.node 24071 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1949_ordered : SortedNatTree.ordered row1949 = true := by decide
set_option maxRecDepth 4096 in
theorem row1949_values : SortedNatTree.values row1949 = Data.row1949 := rfl
theorem row1949_checked : SortedNatTree.contains row1949 = Data.row1949.contains := by
  rw [SortedNatTree.contains_eq_list row1949 row1949_ordered, row1949_values]
def row1951 : SortedNatTree.Tree := (.node 10429 (.node 6151 (.node 4549 (.node 3547 (.node 2683 (.node 2341 .empty .empty) .empty) (.node 3733 (.node 3643 .empty .empty) .empty)) (.node 5227 (.node 4909 (.node 4783 .empty .empty) .empty) (.node 6037 (.node 5437 .empty .empty) .empty))) (.node 8167 (.node 6823 (.node 6547 (.node 6379 .empty .empty) .empty) (.node 7489 (.node 7027 .empty .empty) .empty)) (.node 9421 (.node 8731 (.node 8581 .empty .empty) .empty) (.node 9901 (.node 9883 .empty .empty) .empty)))) (.node 17383 (.node 13687 (.node 11863 (.node 11443 (.node 10939 .empty .empty) .empty) (.node 12301 (.node 12277 .empty .empty) .empty)) (.node 15031 (.node 14149 (.node 13903 .empty .empty) .empty) (.node 17011 (.node 16087 .empty .empty) .empty))) (.node 20047 (.node 18541 (.node 18457 (.node 17599 .empty .empty) .empty) (.node 19237 (.node 19087 .empty .empty) .empty)) (.node 22303 (.node 22171 (.node 21187 .empty .empty) .empty) (.node 22483 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1951_ordered : SortedNatTree.ordered row1951 = true := by decide
set_option maxRecDepth 4096 in
theorem row1951_values : SortedNatTree.values row1951 = Data.row1951 := rfl
theorem row1951_checked : SortedNatTree.contains row1951 = Data.row1951.contains := by
  rw [SortedNatTree.contains_eq_list row1951 row1951_ordered, row1951_values]
def row1973 : SortedNatTree.Tree := (.node 11159 (.node 6131 (.node 4493 (.node 4397 (.node 2381 (.node 1979 .empty .empty) (.node 2591 .empty .empty)) (.node 4457 (.node 4409 .empty .empty) .empty)) (.node 5711 (.node 4919 (.node 4751 .empty .empty) .empty) (.node 5897 (.node 5741 .empty .empty) .empty))) (.node 10181 (.node 8573 (.node 8291 (.node 7547 .empty .empty) (.node 8363 .empty .empty)) (.node 9029 (.node 8837 .empty .empty) .empty)) (.node 10781 (.node 10499 (.node 10433 .empty .empty) .empty) (.node 11057 (.node 10937 .empty .empty) .empty)))) (.node 18917 (.node 15233 (.node 13901 (.node 13121 (.node 12809 .empty .empty) (.node 13397 .empty .empty)) (.node 14771 (.node 14759 .empty .empty) .empty)) (.node 17333 (.node 17231 (.node 16451 .empty .empty) .empty) (.node 18731 (.node 17987 .empty .empty) .empty))) (.node 21647 (.node 20759 (.node 20441 (.node 20399 .empty .empty) .empty) (.node 21563 (.node 21419 .empty .empty) .empty)) (.node 22109 (.node 22067 (.node 21881 .empty .empty) .empty) (.node 23993 (.node 23663 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1973_ordered : SortedNatTree.ordered row1973 = true := by decide
set_option maxRecDepth 4096 in
theorem row1973_values : SortedNatTree.values row1973 = Data.row1973 := rfl
theorem row1973_checked : SortedNatTree.contains row1973 = Data.row1973.contains := by
  rw [SortedNatTree.contains_eq_list row1973 row1973_ordered, row1973_values]
def row1979 : SortedNatTree.Tree := (.node 13187 (.node 8297 (.node 4241 (.node 2693 (.node 2207 (.node 2039 .empty .empty) .empty) (.node 3911 (.node 2999 .empty .empty) .empty)) (.node 7187 (.node 6869 (.node 4937 .empty .empty) .empty) (.node 8291 (.node 7433 .empty .empty) .empty))) (.node 11117 (.node 9413 (.node 9323 (.node 9137 .empty .empty) .empty) (.node 9833 (.node 9491 .empty .empty) .empty)) (.node 12647 (.node 12437 (.node 11279 .empty .empty) .empty) (.node 13049 (.node 12941 .empty .empty) .empty)))) (.node 18047 (.node 14537 (.node 13721 (.node 13523 (.node 13217 .empty .empty) .empty) (.node 14489 (.node 14303 .empty .empty) .empty)) (.node 16661 (.node 15083 (.node 14957 .empty .empty) .empty) (.node 17957 (.node 17669 .empty .empty) .empty))) (.node 22037 (.node 19949 (.node 19157 (.node 18731 .empty .empty) .empty) (.node 21377 (.node 20249 .empty .empty) .empty)) (.node 23021 (.node 22469 (.node 22367 .empty .empty) .empty) (.node 24029 (.node 23549 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1979_ordered : SortedNatTree.ordered row1979 = true := by decide
set_option maxRecDepth 4096 in
theorem row1979_values : SortedNatTree.values row1979 = Data.row1979 := rfl
theorem row1979_checked : SortedNatTree.contains row1979 = Data.row1979.contains := by
  rw [SortedNatTree.contains_eq_list row1979 row1979_ordered, row1979_values]
def row1987 : SortedNatTree.Tree := (.node 8821 (.node 4639 (.node 3331 (.node 2551 (.node 2383 (.node 2131 .empty .empty) (.node 2467 .empty .empty)) (.node 2719 (.node 2707 .empty .empty) (.node 3079 .empty .empty))) (.node 3709 (.node 3499 (.node 3361 .empty .empty) (.node 3517 .empty .empty)) (.node 4423 (.node 3877 .empty .empty) .empty))) (.node 6481 (.node 5659 (.node 5521 (.node 4801 .empty .empty) (.node 5563 .empty .empty)) (.node 6133 (.node 6079 .empty .empty) .empty)) (.node 8191 (.node 7237 (.node 7027 .empty .empty) (.node 7549 .empty .empty)) (.node 8539 (.node 8263 .empty .empty) .empty)))) (.node 16741 (.node 11287 (.node 10513 (.node 9421 (.node 8893 .empty .empty) (.node 9439 .empty .empty)) (.node 10903 (.node 10627 .empty .empty) (.node 11131 .empty .empty))) (.node 13093 (.node 12373 (.node 11353 .empty .empty) (.node 12601 .empty .empty)) (.node 15817 (.node 14143 .empty .empty) .empty))) (.node 19429 (.node 17683 (.node 16831 (.node 16747 .empty .empty) (.node 17293 .empty .empty)) (.node 18973 (.node 17977 .empty .empty) .empty)) (.node 23017 (.node 19891 (.node 19447 .empty .empty) (.node 21673 .empty .empty)) (.node 23773 (.node 23719 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row1987_ordered : SortedNatTree.ordered row1987 = true := by decide
set_option maxRecDepth 4096 in
theorem row1987_values : SortedNatTree.values row1987 = Data.row1987 := rfl
theorem row1987_checked : SortedNatTree.contains row1987 = Data.row1987.contains := by
  rw [SortedNatTree.contains_eq_list row1987 row1987_ordered, row1987_values]
def row1993 : SortedNatTree.Tree := (.node 12049 (.node 6337 (.node 5011 (.node 3457 (.node 3271 (.node 2221 .empty .empty) .empty) (.node 4483 (.node 4093 .empty .empty) .empty)) (.node 5869 (.node 5743 (.node 5581 .empty .empty) .empty) (.node 6211 (.node 6151 .empty .empty) .empty))) (.node 8377 (.node 7507 (.node 6781 (.node 6481 .empty .empty) .empty) (.node 8287 (.node 8209 .empty .empty) .empty)) (.node 11863 (.node 9973 (.node 9277 .empty .empty) .empty) (.node 11941 .empty .empty)))) (.node 18919 (.node 15601 (.node 13063 (.node 12823 (.node 12577 .empty .empty) .empty) (.node 15241 (.node 14197 .empty .empty) .empty)) (.node 17431 (.node 16477 (.node 16063 .empty .empty) .empty) (.node 18439 (.node 17491 .empty .empty) .empty))) (.node 21169 (.node 20071 (.node 19993 (.node 19531 .empty .empty) .empty) (.node 21139 (.node 21061 .empty .empty) .empty)) (.node 22639 (.node 22147 (.node 22003 .empty .empty) .empty) (.node 23563 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1993_ordered : SortedNatTree.ordered row1993 = true := by decide
set_option maxRecDepth 4096 in
theorem row1993_values : SortedNatTree.values row1993 = Data.row1993 := rfl
theorem row1993_checked : SortedNatTree.contains row1993 = Data.row1993.contains := by
  rw [SortedNatTree.contains_eq_list row1993 row1993_ordered, row1993_values]
def row1997 : SortedNatTree.Tree := (.node 12437 (.node 7043 (.node 3677 (.node 2903 (.node 2423 (.node 2003 .empty .empty) .empty) (.node 3323 (.node 3119 .empty .empty) .empty)) (.node 4793 (.node 4583 (.node 4397 .empty .empty) .empty) (.node 6053 .empty .empty))) (.node 9941 (.node 9413 (.node 8831 (.node 7583 .empty .empty) .empty) (.node 9473 .empty .empty)) (.node 11549 (.node 11117 (.node 10163 .empty .empty) .empty) (.node 12377 .empty .empty)))) (.node 18149 (.node 15773 (.node 13109 (.node 12713 (.node 12671 .empty .empty) .empty) (.node 14951 (.node 14153 .empty .empty) .empty)) (.node 17681 (.node 16001 (.node 15959 .empty .empty) .empty) (.node 18143 .empty .empty))) (.node 21107 (.node 19853 (.node 19301 (.node 19073 .empty .empty) .empty) (.node 20441 .empty .empty)) (.node 23399 (.node 23057 (.node 21383 .empty .empty) .empty) (.node 23417 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1997_ordered : SortedNatTree.ordered row1997 = true := by decide
set_option maxRecDepth 4096 in
theorem row1997_values : SortedNatTree.values row1997 = Data.row1997 := rfl
theorem row1997_checked : SortedNatTree.contains row1997 = Data.row1997.contains := by
  rw [SortedNatTree.contains_eq_list row1997 row1997_ordered, row1997_values]
def row1999 : SortedNatTree.Tree := (.node 13267 (.node 7717 (.node 3733 (.node 3361 (.node 2521 (.node 2347 .empty .empty) .empty) (.node 3607 (.node 3469 .empty .empty) .empty)) (.node 6469 (.node 6121 (.node 5869 .empty .empty) .empty) (.node 6961 .empty .empty))) (.node 10657 (.node 9739 (.node 8779 (.node 8263 .empty .empty) .empty) (.node 10567 .empty .empty)) (.node 12421 (.node 12379 (.node 11149 .empty .empty) .empty) (.node 12541 .empty .empty)))) (.node 19309 (.node 16231 (.node 14563 (.node 14557 (.node 14437 .empty .empty) .empty) (.node 16033 (.node 15787 .empty .empty) .empty)) (.node 17299 (.node 16747 (.node 16567 .empty .empty) .empty) (.node 18679 .empty .empty))) (.node 22291 (.node 21031 (.node 19483 (.node 19441 .empty .empty) .empty) (.node 21193 .empty .empty)) (.node 23599 (.node 22501 (.node 22369 .empty .empty) .empty) (.node 23671 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row1999_ordered : SortedNatTree.ordered row1999 = true := by decide
set_option maxRecDepth 4096 in
theorem row1999_values : SortedNatTree.values row1999 = Data.row1999 := rfl
theorem row1999_checked : SortedNatTree.contains row1999 = Data.row1999.contains := by
  rw [SortedNatTree.contains_eq_list row1999 row1999_ordered, row1999_values]
def row2003 : SortedNatTree.Tree := (.node 10091 (.node 5693 (.node 4673 (.node 3413 (.node 2237 (.node 2087 .empty .empty) (.node 2543 .empty .empty)) (.node 4271 (.node 3557 .empty .empty) .empty)) (.node 5189 (.node 5051 (.node 5039 .empty .empty) (.node 5153 .empty .empty)) (.node 5657 (.node 5651 .empty .empty) .empty))) (.node 7211 (.node 6833 (.node 6689 (.node 6563 .empty .empty) (.node 6791 .empty .empty)) (.node 7013 (.node 6983 .empty .empty) .empty)) (.node 8297 (.node 8273 (.node 8171 .empty .empty) .empty) (.node 8747 (.node 8609 .empty .empty) .empty)))) (.node 15683 (.node 12713 (.node 10973 (.node 10601 (.node 10343 .empty .empty) (.node 10739 .empty .empty)) (.node 11831 (.node 11777 .empty .empty) .empty)) (.node 14669 (.node 13721 (.node 13313 .empty .empty) (.node 13931 .empty .empty)) (.node 15329 (.node 14867 .empty .empty) .empty))) (.node 19583 (.node 17573 (.node 16607 (.node 15959 .empty .empty) (.node 16673 .empty .empty)) (.node 18749 (.node 17597 .empty .empty) .empty)) (.node 21419 (.node 20693 (.node 19937 .empty .empty) .empty) (.node 22541 (.node 22259 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2003_ordered : SortedNatTree.ordered row2003 = true := by decide
set_option maxRecDepth 4096 in
theorem row2003_values : SortedNatTree.values row2003 = Data.row2003 := rfl
theorem row2003_checked : SortedNatTree.contains row2003 = Data.row2003.contains := by
  rw [SortedNatTree.contains_eq_list row2003 row2003_ordered, row2003_values]
def row2011 : SortedNatTree.Tree := (.node 13183 (.node 6781 (.node 4177 (.node 4093 (.node 3673 (.node 3253 .empty .empty) .empty) (.node 4099 .empty .empty)) (.node 6379 (.node 5521 (.node 5413 .empty .empty) .empty) (.node 6637 .empty .empty))) (.node 9433 (.node 9049 (.node 8419 (.node 7963 .empty .empty) .empty) (.node 9091 .empty .empty)) (.node 10477 (.node 9769 (.node 9661 .empty .empty) .empty) (.node 12409 .empty .empty)))) (.node 18661 (.node 15877 (.node 15331 (.node 14923 (.node 14293 .empty .empty) .empty) (.node 15349 .empty .empty)) (.node 17047 (.node 16693 (.node 16063 .empty .empty) .empty) (.node 18481 .empty .empty))) (.node 20749 (.node 20287 (.node 20113 (.node 19423 .empty .empty) .empty) (.node 20323 .empty .empty)) (.node 22573 (.node 22093 (.node 21673 .empty .empty) .empty) (.node 23269 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2011_ordered : SortedNatTree.ordered row2011 = true := by decide
set_option maxRecDepth 4096 in
theorem row2011_values : SortedNatTree.values row2011 = Data.row2011 := rfl
theorem row2011_checked : SortedNatTree.contains row2011 = Data.row2011.contains := by
  rw [SortedNatTree.contains_eq_list row2011 row2011_ordered, row2011_values]
def row2017 : SortedNatTree.Tree := (.node 10651 (.node 6967 (.node 4339 (.node 2689 (.node 2617 (.node 2221 .empty .empty) .empty) (.node 3919 (.node 3313 .empty .empty) .empty)) (.node 5653 (.node 4621 (.node 4423 .empty .empty) .empty) (.node 6661 .empty .empty))) (.node 8707 (.node 8311 (.node 7603 (.node 7237 .empty .empty) .empty) (.node 8467 (.node 8317 .empty .empty) .empty)) (.node 9739 (.node 9001 (.node 8971 .empty .empty) .empty) (.node 9787 .empty .empty)))) (.node 18973 (.node 12919 (.node 11197 (.node 10987 (.node 10711 .empty .empty) .empty) (.node 12697 (.node 11299 .empty .empty) .empty)) (.node 18061 (.node 17977 (.node 16087 .empty .empty) .empty) (.node 18367 .empty .empty))) (.node 21283 (.node 19753 (.node 19681 (.node 19489 .empty .empty) .empty) (.node 20593 (.node 20047 .empty .empty) .empty)) (.node 23251 (.node 23143 (.node 21319 .empty .empty) .empty) (.node 23689 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2017_ordered : SortedNatTree.ordered row2017 = true := by decide
set_option maxRecDepth 4096 in
theorem row2017_values : SortedNatTree.values row2017 = Data.row2017 := rfl
theorem row2017_checked : SortedNatTree.contains row2017 = Data.row2017.contains := by
  rw [SortedNatTree.contains_eq_list row2017 row2017_ordered, row2017_values]
def row2027 : SortedNatTree.Tree := (.node 12611 (.node 7457 (.node 5483 (.node 4259 (.node 3761 (.node 3119 .empty .empty) .empty) (.node 5153 .empty .empty)) (.node 6911 (.node 6701 (.node 6203 .empty .empty) .empty) (.node 7253 .empty .empty))) (.node 10181 (.node 9803 (.node 8837 (.node 7703 .empty .empty) .empty) (.node 9839 .empty .empty)) (.node 10889 (.node 10631 (.node 10463 .empty .empty) .empty) (.node 12149 .empty .empty)))) (.node 18947 (.node 15473 (.node 15263 (.node 13781 (.node 12791 .empty .empty) .empty) (.node 15269 .empty .empty)) (.node 18077 (.node 17483 (.node 16433 .empty .empty) .empty) (.node 18401 .empty .empty))) (.node 21863 (.node 20753 (.node 19793 (.node 19763 .empty .empty) .empty) (.node 21191 .empty .empty)) (.node 23063 (.node 22853 (.node 22067 .empty .empty) .empty) (.node 23753 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2027_ordered : SortedNatTree.ordered row2027 = true := by decide
set_option maxRecDepth 4096 in
theorem row2027_values : SortedNatTree.values row2027 = Data.row2027 := rfl
theorem row2027_checked : SortedNatTree.contains row2027 = Data.row2027.contains := by
  rw [SortedNatTree.contains_eq_list row2027 row2027_ordered, row2027_values]
def row2029 : SortedNatTree.Tree := (.node 11047 (.node 5701 (.node 5101 (.node 2791 (.node 2341 (.node 2281 .empty .empty) .empty) (.node 4507 (.node 3739 .empty .empty) .empty)) (.node 5419 (.node 5281 (.node 5227 .empty .empty) .empty) (.node 5689 .empty .empty))) (.node 8293 (.node 7027 (.node 6247 (.node 6121 .empty .empty) .empty) (.node 7309 (.node 7213 .empty .empty) .empty)) (.node 9859 (.node 9679 (.node 9439 .empty .empty) .empty) (.node 9931 .empty .empty)))) (.node 15907 (.node 12109 (.node 11299 (.node 11287 (.node 11257 .empty .empty) .empty) (.node 11827 (.node 11701 .empty .empty) .empty)) (.node 14197 (.node 13297 (.node 12907 .empty .empty) .empty) (.node 14323 .empty .empty))) (.node 19681 (.node 18253 (.node 17977 (.node 17581 .empty .empty) .empty) (.node 19471 (.node 19009 .empty .empty) .empty)) (.node 23269 (.node 23197 (.node 22861 .empty .empty) .empty) (.node 23311 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2029_ordered : SortedNatTree.ordered row2029 = true := by decide
set_option maxRecDepth 4096 in
theorem row2029_values : SortedNatTree.values row2029 = Data.row2029 := rfl
theorem row2029_checked : SortedNatTree.contains row2029 = Data.row2029.contains := by
  rw [SortedNatTree.contains_eq_list row2029 row2029_ordered, row2029_values]
def row2039 : SortedNatTree.Tree := (.node 11069 (.node 4799 (.node 3491 (.node 2591 (.node 2417 (.node 2357 .empty .empty) .empty) (.node 3323 (.node 3299 .empty .empty) .empty)) (.node 4079 (.node 3917 (.node 3767 .empty .empty) .empty) (.node 4523 .empty .empty))) (.node 7673 (.node 6203 (.node 6053 (.node 4937 .empty .empty) .empty) (.node 6689 .empty .empty)) (.node 10331 (.node 10253 (.node 10079 .empty .empty) .empty) (.node 10463 .empty .empty)))) (.node 18503 (.node 15749 (.node 15329 (.node 14891 (.node 12941 .empty .empty) .empty) (.node 15461 .empty .empty)) (.node 16829 (.node 16553 (.node 16007 .empty .empty) .empty) (.node 18269 .empty .empty))) (.node 21587 (.node 20021 (.node 19181 (.node 18593 .empty .empty) .empty) (.node 20393 .empty .empty)) (.node 22091 (.node 22031 (.node 22013 .empty .empty) .empty) (.node 22409 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2039_ordered : SortedNatTree.ordered row2039 = true := by decide
set_option maxRecDepth 4096 in
theorem row2039_values : SortedNatTree.values row2039 = Data.row2039 := rfl
theorem row2039_checked : SortedNatTree.contains row2039 = Data.row2039.contains := by
  rw [SortedNatTree.contains_eq_list row2039 row2039_ordered, row2039_values]
def row2053 : SortedNatTree.Tree := (.node 14419 (.node 6421 (.node 4783 (.node 3361 (.node 3049 (.node 2797 .empty .empty) .empty) (.node 4729 .empty .empty)) (.node 6343 (.node 5653 .empty .empty) (.node 6361 .empty .empty))) (.node 12097 (.node 8521 (.node 8317 (.node 6619 .empty .empty) .empty) (.node 8923 .empty .empty)) (.node 13627 (.node 12781 .empty .empty) (.node 13921 .empty .empty)))) (.node 18181 (.node 16249 (.node 15937 (.node 15889 (.node 14929 .empty .empty) .empty) (.node 16087 .empty .empty)) (.node 17191 (.node 16879 .empty .empty) (.node 17257 .empty .empty))) (.node 22111 (.node 19069 (.node 18787 (.node 18523 .empty .empty) .empty) (.node 21799 .empty .empty)) (.node 23143 (.node 23029 .empty .empty) (.node 23497 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2053_ordered : SortedNatTree.ordered row2053 = true := by decide
set_option maxRecDepth 4096 in
theorem row2053_values : SortedNatTree.values row2053 = Data.row2053 := rfl
theorem row2053_checked : SortedNatTree.contains row2053 = Data.row2053.contains := by
  rw [SortedNatTree.contains_eq_list row2053 row2053_ordered, row2053_values]
def row2063 : SortedNatTree.Tree := (.node 11579 (.node 5477 (.node 3617 (.node 2333 (.node 2243 (.node 2099 .empty .empty) .empty) (.node 3089 (.node 2411 .empty .empty) .empty)) (.node 4349 (.node 4217 (.node 3803 .empty .empty) .empty) (.node 5231 .empty .empty))) (.node 6599 (.node 5981 (.node 5903 (.node 5711 .empty .empty) .empty) (.node 6449 .empty .empty)) (.node 9239 (.node 8861 (.node 8741 .empty .empty) .empty) (.node 9791 .empty .empty)))) (.node 17597 (.node 16529 (.node 13697 (.node 13109 (.node 12101 .empty .empty) .empty) (.node 14621 (.node 14177 .empty .empty) .empty)) (.node 16889 (.node 16673 (.node 16631 .empty .empty) .empty) (.node 17387 .empty .empty))) (.node 19979 (.node 19391 (.node 18719 (.node 18521 .empty .empty) .empty) (.node 19697 .empty .empty)) (.node 22157 (.node 21737 (.node 21383 .empty .empty) .empty) (.node 23459 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2063_ordered : SortedNatTree.ordered row2063 = true := by decide
set_option maxRecDepth 4096 in
theorem row2063_values : SortedNatTree.values row2063 = Data.row2063 := rfl
theorem row2063_checked : SortedNatTree.contains row2063 = Data.row2063.contains := by
  rw [SortedNatTree.contains_eq_list row2063 row2063_ordered, row2063_values]
def row2069 : SortedNatTree.Tree := (.node 13799 (.node 7283 (.node 5021 (.node 4397 (.node 2339 (.node 2297 .empty .empty) .empty) (.node 4691 (.node 4637 .empty .empty) .empty)) (.node 6983 (.node 6959 (.node 6827 .empty .empty) .empty) (.node 7253 (.node 7121 .empty .empty) .empty))) (.node 9059 (.node 8219 (.node 7877 (.node 7757 .empty .empty) .empty) (.node 8747 (.node 8369 .empty .empty) .empty)) (.node 11069 (.node 10433 (.node 9221 .empty .empty) .empty) (.node 12239 (.node 11093 .empty .empty) .empty)))) (.node 17573 (.node 14723 (.node 14207 (.node 14033 (.node 13907 .empty .empty) .empty) (.node 14627 (.node 14423 .empty .empty) .empty)) (.node 15461 (.node 15017 (.node 14759 .empty .empty) .empty) (.node 16007 (.node 15887 .empty .empty) .empty))) (.node 21269 (.node 20183 (.node 19463 (.node 18077 .empty .empty) .empty) (.node 21017 (.node 20807 .empty .empty) .empty)) (.node 22943 (.node 22739 (.node 22481 .empty .empty) .empty) (.node 23669 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2069_ordered : SortedNatTree.ordered row2069 = true := by decide
set_option maxRecDepth 4096 in
theorem row2069_values : SortedNatTree.values row2069 = Data.row2069 := rfl
theorem row2069_checked : SortedNatTree.contains row2069 = Data.row2069.contains := by
  rw [SortedNatTree.contains_eq_list row2069 row2069_ordered, row2069_values]
def row2081 : SortedNatTree.Tree := (.node 11633 (.node 6791 (.node 5153 (.node 4349 (.node 3323 (.node 2699 .empty .empty) (.node 4127 .empty .empty)) (.node 4943 (.node 4691 .empty .empty) .empty)) (.node 5801 (.node 5657 (.node 5441 .empty .empty) .empty) (.node 6689 (.node 5807 .empty .empty) .empty))) (.node 9629 (.node 7793 (.node 7229 (.node 7109 .empty .empty) .empty) (.node 9371 (.node 8747 .empty .empty) .empty)) (.node 10691 (.node 10289 (.node 9857 .empty .empty) .empty) (.node 11489 (.node 10733 .empty .empty) .empty)))) (.node 18461 (.node 14957 (.node 13523 (.node 12713 (.node 12473 .empty .empty) (.node 13487 .empty .empty)) (.node 14891 (.node 13799 .empty .empty) .empty)) (.node 18041 (.node 17207 (.node 15329 .empty .empty) .empty) (.node 18341 (.node 18149 .empty .empty) .empty))) (.node 20981 (.node 19991 (.node 19553 (.node 18671 .empty .empty) .empty) (.node 20441 (.node 20333 .empty .empty) .empty)) (.node 22877 (.node 22643 (.node 21713 .empty .empty) .empty) (.node 23291 (.node 22973 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2081_ordered : SortedNatTree.ordered row2081 = true := by decide
set_option maxRecDepth 4096 in
theorem row2081_values : SortedNatTree.values row2081 = Data.row2081 := rfl
theorem row2081_checked : SortedNatTree.contains row2081 = Data.row2081.contains := by
  rw [SortedNatTree.contains_eq_list row2081 row2081_ordered, row2081_values]
def row2083 : SortedNatTree.Tree := (.node 12301 (.node 8221 (.node 5443 (.node 4111 (.node 3343 (.node 2971 .empty .empty) (.node 3511 .empty .empty)) (.node 4651 (.node 4243 .empty .empty) .empty)) (.node 6121 (.node 5923 (.node 5827 .empty .empty) .empty) (.node 7459 (.node 6997 .empty .empty) .empty))) (.node 9901 (.node 8929 (.node 8677 (.node 8599 .empty .empty) .empty) (.node 9829 (.node 9109 .empty .empty) .empty)) (.node 11287 (.node 11197 (.node 10939 .empty .empty) .empty) (.node 11827 (.node 11617 .empty .empty) .empty)))) (.node 18301 (.node 14929 (.node 13297 (.node 12619 (.node 12451 .empty .empty) .empty) (.node 14851 (.node 14149 .empty .empty) .empty)) (.node 17581 (.node 17293 (.node 15361 .empty .empty) .empty) (.node 18013 (.node 17911 .empty .empty) .empty))) (.node 20443 (.node 18541 (.node 18457 (.node 18307 .empty .empty) .empty) (.node 20107 (.node 19273 .empty .empty) .empty)) (.node 22717 (.node 22129 (.node 21991 .empty .empty) .empty) (.node 23497 (.node 23473 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2083_ordered : SortedNatTree.ordered row2083 = true := by decide
set_option maxRecDepth 4096 in
theorem row2083_values : SortedNatTree.values row2083 = Data.row2083 := rfl
theorem row2083_checked : SortedNatTree.contains row2083 = Data.row2083.contains := by
  rw [SortedNatTree.contains_eq_list row2083 row2083_ordered, row2083_values]
def row2087 : SortedNatTree.Tree := (.node 14543 (.node 7559 (.node 4937 (.node 3821 (.node 2417 (.node 2153 .empty .empty) .empty) (.node 4463 .empty .empty)) (.node 6689 (.node 5333 (.node 5009 .empty .empty) .empty) (.node 7079 .empty .empty))) (.node 9437 (.node 8969 (.node 8537 (.node 7907 .empty .empty) .empty) (.node 9029 .empty .empty)) (.node 12011 (.node 11831 .empty .empty) (.node 13217 .empty .empty)))) (.node 19079 (.node 16229 (.node 15473 (.node 15401 (.node 14657 .empty .empty) .empty) (.node 15971 .empty .empty)) (.node 17627 (.node 17291 (.node 17231 .empty .empty) .empty) (.node 17837 .empty .empty))) (.node 23003 (.node 21059 (.node 20411 (.node 19391 .empty .empty) .empty) (.node 22511 .empty .empty)) (.node 23561 (.node 23027 .empty .empty) (.node 23633 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2087_ordered : SortedNatTree.ordered row2087 = true := by decide
set_option maxRecDepth 4096 in
theorem row2087_values : SortedNatTree.values row2087 = Data.row2087 := rfl
theorem row2087_checked : SortedNatTree.contains row2087 = Data.row2087.contains := by
  rw [SortedNatTree.contains_eq_list row2087 row2087_ordered, row2087_values]
def row2089 : SortedNatTree.Tree := (.node 9679 (.node 6277 (.node 4099 (.node 3181 (.node 2953 (.node 2341 .empty .empty) .empty) (.node 3931 (.node 3643 .empty .empty) .empty)) (.node 5443 (.node 4597 (.node 4219 .empty .empty) .empty) (.node 5827 .empty .empty))) (.node 7873 (.node 7297 (.node 7237 (.node 7039 .empty .empty) .empty) (.node 7411 .empty .empty)) (.node 9049 (.node 8263 (.node 8209 .empty .empty) .empty) (.node 9091 .empty .empty)))) (.node 15859 (.node 13249 (.node 12037 (.node 10687 (.node 10369 .empty .empty) .empty) (.node 12757 .empty .empty)) (.node 15277 (.node 13921 (.node 13633 .empty .empty) .empty) (.node 15823 .empty .empty))) (.node 19507 (.node 18121 (.node 17041 (.node 16693 .empty .empty) .empty) (.node 18181 .empty .empty)) (.node 22453 (.node 21997 (.node 20533 .empty .empty) .empty) (.node 23041 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2089_ordered : SortedNatTree.ordered row2089 = true := by decide
set_option maxRecDepth 4096 in
theorem row2089_values : SortedNatTree.values row2089 = Data.row2089 := rfl
theorem row2089_checked : SortedNatTree.contains row2089 = Data.row2089.contains := by
  rw [SortedNatTree.contains_eq_list row2089 row2089_ordered, row2089_values]
def row2099 : SortedNatTree.Tree := (.node 9887 (.node 5279 (.node 2843 (.node 2393 (.node 2381 (.node 2129 .empty .empty) .empty) (.node 2693 .empty .empty)) (.node 3527 (.node 3083 .empty .empty) (.node 4241 .empty .empty))) (.node 7841 (.node 7211 (.node 6863 (.node 5717 .empty .empty) .empty) (.node 7517 .empty .empty)) (.node 8093 (.node 8069 .empty .empty) (.node 9479 .empty .empty)))) (.node 17783 (.node 14591 (.node 12473 (.node 10559 (.node 10343 .empty .empty) .empty) (.node 12953 .empty .empty)) (.node 14879 (.node 14867 .empty .empty) (.node 17597 .empty .empty))) (.node 21059 (.node 19211 (.node 18803 .empty .empty) (.node 19793 .empty .empty)) (.node 22907 (.node 21143 .empty .empty) (.node 23159 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2099_ordered : SortedNatTree.ordered row2099 = true := by decide
set_option maxRecDepth 4096 in
theorem row2099_values : SortedNatTree.values row2099 = Data.row2099 := rfl
theorem row2099_checked : SortedNatTree.contains row2099 = Data.row2099.contains := by
  rw [SortedNatTree.contains_eq_list row2099 row2099_ordered, row2099_values]
def row2111 : SortedNatTree.Tree := (.node 10061 (.node 4229 (.node 3251 (.node 3041 (.node 2969 (.node 2411 .empty .empty) (.node 2999 .empty .empty)) (.node 3221 (.node 3083 .empty .empty) .empty)) (.node 4019 (.node 3467 (.node 3329 .empty .empty) .empty) (.node 4217 (.node 4127 .empty .empty) .empty))) (.node 7487 (.node 5261 (.node 4931 (.node 4649 .empty .empty) .empty) (.node 6911 (.node 6857 .empty .empty) .empty)) (.node 8753 (.node 8219 (.node 7793 .empty .empty) .empty) (.node 9257 (.node 8849 .empty .empty) .empty)))) (.node 17957 (.node 13577 (.node 11399 (.node 10733 (.node 10313 .empty .empty) .empty) (.node 12437 (.node 11939 .empty .empty) .empty)) (.node 15803 (.node 14843 (.node 14747 .empty .empty) .empty) (.node 17321 (.node 15881 .empty .empty) .empty))) (.node 22277 (.node 20771 (.node 20759 (.node 19289 .empty .empty) .empty) (.node 21557 (.node 21107 .empty .empty) .empty)) (.node 23789 (.node 23291 (.node 22283 .empty .empty) .empty) (.node 23909 (.node 23879 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2111_ordered : SortedNatTree.ordered row2111 = true := by decide
set_option maxRecDepth 4096 in
theorem row2111_values : SortedNatTree.values row2111 = Data.row2111 := rfl
theorem row2111_checked : SortedNatTree.contains row2111 = Data.row2111.contains := by
  rw [SortedNatTree.contains_eq_list row2111 row2111_ordered, row2111_values]
def row2113 : SortedNatTree.Tree := (.node 9649 (.node 6067 (.node 3727 (.node 3121 (.node 2659 (.node 2131 .empty .empty) .empty) (.node 3583 (.node 3433 .empty .empty) .empty)) (.node 5227 (.node 4507 (.node 4093 .empty .empty) .empty) (.node 5431 (.node 5419 .empty .empty) .empty))) (.node 7927 (.node 6553 (.node 6481 (.node 6277 .empty .empty) .empty) (.node 7393 (.node 7309 .empty .empty) .empty)) (.node 9337 (.node 9127 (.node 8713 .empty .empty) .empty) (.node 9439 .empty .empty)))) (.node 16453 (.node 12097 (.node 11239 (.node 10711 (.node 10513 .empty .empty) .empty) (.node 11497 (.node 11467 .empty .empty) .empty)) (.node 15583 (.node 14461 (.node 12919 .empty .empty) .empty) (.node 16381 (.node 15733 .empty .empty) .empty))) (.node 21379 (.node 19813 (.node 18973 (.node 18451 .empty .empty) .empty) (.node 21157 (.node 21139 .empty .empty) .empty)) (.node 22621 (.node 22369 (.node 22303 .empty .empty) .empty) (.node 23911 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2113_ordered : SortedNatTree.ordered row2113 = true := by decide
set_option maxRecDepth 4096 in
theorem row2113_values : SortedNatTree.values row2113 = Data.row2113 := rfl
theorem row2113_checked : SortedNatTree.contains row2113 = Data.row2113.contains := by
  rw [SortedNatTree.contains_eq_list row2113 row2113_ordered, row2113_values]
def row2129 : SortedNatTree.Tree := (.node 11969 (.node 6491 (.node 4517 (.node 3947 (.node 2729 (.node 2423 .empty .empty) (.node 3089 .empty .empty)) (.node 4271 (.node 4001 .empty .empty) .empty)) (.node 5783 (.node 5051 (.node 4937 .empty .empty) .empty) (.node 6269 (.node 6203 .empty .empty) .empty))) (.node 7949 (.node 7247 (.node 7121 (.node 6791 .empty .empty) (.node 7193 .empty .empty)) (.node 7937 (.node 7559 .empty .empty) .empty)) (.node 10259 (.node 9029 (.node 8999 .empty .empty) .empty) (.node 10937 (.node 10457 .empty .empty) .empty)))) (.node 19037 (.node 15641 (.node 13751 (.node 13163 (.node 12263 .empty .empty) (.node 13229 .empty .empty)) (.node 14759 (.node 14087 .empty .empty) .empty)) (.node 18059 (.node 16547 (.node 16481 .empty .empty) .empty) (.node 19001 (.node 18341 .empty .empty) .empty))) (.node 21407 (.node 20507 (.node 19979 (.node 19697 .empty .empty) (.node 20051 .empty .empty)) (.node 21221 (.node 21179 .empty .empty) .empty)) (.node 23327 (.node 21563 (.node 21491 .empty .empty) .empty) (.node 23813 (.node 23549 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2129_ordered : SortedNatTree.ordered row2129 = true := by decide
set_option maxRecDepth 4096 in
theorem row2129_values : SortedNatTree.values row2129 = Data.row2129 := rfl
theorem row2129_checked : SortedNatTree.contains row2129 = Data.row2129.contains := by
  rw [SortedNatTree.contains_eq_list row2129 row2129_ordered, row2129_values]
def row2131 : SortedNatTree.Tree := (.node 10159 (.node 5563 (.node 3541 (.node 2833 (.node 2677 (.node 2293 .empty .empty) .empty) (.node 3343 (.node 3037 .empty .empty) .empty)) (.node 4567 (.node 4297 (.node 3547 .empty .empty) .empty) (.node 5023 (.node 4933 .empty .empty) .empty))) (.node 7873 (.node 7027 (.node 6619 (.node 5689 .empty .empty) .empty) (.node 7669 (.node 7129 .empty .empty) .empty)) (.node 9769 (.node 8839 (.node 8377 .empty .empty) .empty) (.node 9817 .empty .empty)))) (.node 18121 (.node 14173 (.node 12373 (.node 12049 (.node 10273 .empty .empty) .empty) (.node 13339 (.node 12919 .empty .empty) .empty)) (.node 16567 (.node 15619 (.node 14821 .empty .empty) .empty) (.node 18049 (.node 16699 .empty .empty) .empty))) (.node 21661 (.node 19441 (.node 19381 (.node 19267 .empty .empty) .empty) (.node 20563 (.node 19597 .empty .empty) .empty)) (.node 23473 (.node 21943 (.node 21751 .empty .empty) .empty) (.node 23623 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2131_ordered : SortedNatTree.ordered row2131 = true := by decide
set_option maxRecDepth 4096 in
theorem row2131_values : SortedNatTree.values row2131 = Data.row2131 := rfl
theorem row2131_checked : SortedNatTree.contains row2131 = Data.row2131.contains := by
  rw [SortedNatTree.contains_eq_list row2131 row2131_ordered, row2131_values]
def row2137 : SortedNatTree.Tree := (.node 10099 (.node 6991 (.node 4513 (.node 3061 (.node 2539 (.node 2293 .empty .empty) .empty) (.node 4051 (.node 3697 .empty .empty) .empty)) (.node 6133 (.node 6073 (.node 5503 .empty .empty) .empty) (.node 6571 (.node 6217 .empty .empty) .empty))) (.node 8317 (.node 7459 (.node 7159 (.node 7039 .empty .empty) .empty) (.node 8233 (.node 7561 .empty .empty) .empty)) (.node 9157 (.node 8527 (.node 8419 .empty .empty) .empty) (.node 9973 (.node 9283 .empty .empty) .empty)))) (.node 17137 (.node 13063 (.node 11257 (.node 10957 (.node 10597 .empty .empty) .empty) (.node 11779 (.node 11299 .empty .empty) .empty)) (.node 14251 (.node 14173 (.node 13921 .empty .empty) .empty) (.node 16903 (.node 14797 .empty .empty) .empty))) (.node 19501 (.node 18397 (.node 17929 (.node 17341 .empty .empty) .empty) (.node 18787 (.node 18481 .empty .empty) .empty)) (.node 22303 (.node 20407 (.node 19777 .empty .empty) .empty) (.node 23623 (.node 22441 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2137_ordered : SortedNatTree.ordered row2137 = true := by decide
set_option maxRecDepth 4096 in
theorem row2137_values : SortedNatTree.values row2137 = Data.row2137 := rfl
theorem row2137_checked : SortedNatTree.contains row2137 = Data.row2137.contains := by
  rw [SortedNatTree.contains_eq_list row2137 row2137_ordered, row2137_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
