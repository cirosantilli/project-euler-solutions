import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row6421 : SortedNatTree.Tree := (.node 12451 (.node 11299 (.node 9973 (.node 6949 (.node 6553 .empty .empty) (.node 8377 .empty .empty)) (.node 10957 (.node 10477 .empty .empty) .empty)) (.node 11779 (.node 11689 (.node 11467 .empty .empty) .empty) (.node 11923 (.node 11887 .empty .empty) .empty))) (.node 17491 (.node 14593 (.node 12739 (.node 12511 .empty .empty) (.node 14197 .empty .empty)) (.node 16903 (.node 16183 .empty .empty) .empty)) (.node 18517 (.node 18457 (.node 17551 .empty .empty) .empty) (.node 19447 (.node 19381 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6421_ordered : SortedNatTree.ordered row6421 = true := by decide
set_option maxRecDepth 4096 in
theorem row6421_values : SortedNatTree.values row6421 = Data.row6421 := rfl
theorem row6421_checked : SortedNatTree.contains row6421 = Data.row6421.contains := by
  rw [SortedNatTree.contains_eq_list row6421 row6421_ordered, row6421_values]
def row6427 : SortedNatTree.Tree := (.node 13063 (.node 10111 (.node 8233 (.node 7069 (.node 6997 .empty .empty) (.node 8101 .empty .empty)) (.node 8971 (.node 8941 .empty .empty) (.node 9067 .empty .empty))) (.node 12379 (.node 11467 (.node 10837 .empty .empty) (.node 12211 .empty .empty)) (.node 12919 (.node 12589 .empty .empty) .empty))) (.node 15193 (.node 14503 (.node 13933 (.node 13147 .empty .empty) (.node 14221 .empty .empty)) (.node 15091 (.node 14923 .empty .empty) .empty)) (.node 17041 (.node 15511 (.node 15439 .empty .empty) (.node 16741 .empty .empty)) (.node 19387 (.node 18229 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6427_ordered : SortedNatTree.ordered row6427 = true := by decide
set_option maxRecDepth 4096 in
theorem row6427_values : SortedNatTree.values row6427 = Data.row6427 := rfl
theorem row6427_checked : SortedNatTree.contains row6427 = Data.row6427.contains := by
  rw [SortedNatTree.contains_eq_list row6427 row6427_ordered, row6427_values]
def row6449 : SortedNatTree.Tree := (.node 13337 (.node 10193 (.node 8747 (.node 8171 (.node 7523 .empty .empty) (.node 8219 .empty .empty)) (.node 9209 (.node 9203 .empty .empty) (.node 9431 .empty .empty))) (.node 12503 (.node 11279 (.node 10463 .empty .empty) (.node 12011 .empty .empty)) (.node 13151 (.node 12641 .empty .empty) .empty))) (.node 15797 (.node 14867 (.node 14327 (.node 13757 .empty .empty) (.node 14723 .empty .empty)) (.node 15731 (.node 15629 .empty .empty) .empty)) (.node 18143 (.node 16661 (.node 15959 .empty .empty) (.node 17159 .empty .empty)) (.node 19403 (.node 18803 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6449_ordered : SortedNatTree.ordered row6449 = true := by decide
set_option maxRecDepth 4096 in
theorem row6449_values : SortedNatTree.values row6449 = Data.row6449 := rfl
theorem row6449_checked : SortedNatTree.contains row6449 = Data.row6449.contains := by
  rw [SortedNatTree.contains_eq_list row6449 row6449_ordered, row6449_values]
def row6451 : SortedNatTree.Tree := (.node 11197 (.node 9241 (.node 7753 (.node 7309 (.node 7213 .empty .empty) (.node 7561 .empty .empty)) (.node 9187 (.node 8191 .empty .empty) .empty)) (.node 9787 (.node 9781 (.node 9643 .empty .empty) .empty) (.node 10957 (.node 10273 .empty .empty) .empty))) (.node 15271 (.node 14341 (.node 12541 (.node 11923 .empty .empty) (.node 13633 .empty .empty)) (.node 14551 (.node 14479 .empty .empty) .empty)) (.node 18169 (.node 16447 (.node 16273 .empty .empty) .empty) (.node 19183 (.node 18583 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6451_ordered : SortedNatTree.ordered row6451 = true := by decide
set_option maxRecDepth 4096 in
theorem row6451_values : SortedNatTree.values row6451 = Data.row6451 := rfl
theorem row6451_checked : SortedNatTree.contains row6451 = Data.row6451.contains := by
  rw [SortedNatTree.contains_eq_list row6451 row6451_ordered, row6451_values]
def row6469 : SortedNatTree.Tree := (.node 11833 (.node 8233 (.node 6997 (.node 6673 (.node 6481 .empty .empty) (.node 6703 .empty .empty)) (.node 7681 (.node 7219 .empty .empty) (.node 7879 .empty .empty))) (.node 11299 (.node 8863 (.node 8527 .empty .empty) (.node 9241 .empty .empty)) (.node 11827 (.node 11437 .empty .empty) .empty))) (.node 16447 (.node 14461 (.node 11971 (.node 11887 .empty .empty) (.node 12379 .empty .empty)) (.node 15583 (.node 14731 .empty .empty) (.node 16141 .empty .empty))) (.node 18637 (.node 17107 (.node 17077 .empty .empty) (.node 18313 .empty .empty)) (.node 19417 (.node 19267 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6469_ordered : SortedNatTree.ordered row6469 = true := by decide
set_option maxRecDepth 4096 in
theorem row6469_values : SortedNatTree.values row6469 = Data.row6469 := rfl
theorem row6469_checked : SortedNatTree.contains row6469 = Data.row6469.contains := by
  rw [SortedNatTree.contains_eq_list row6469 row6469_ordered, row6469_values]
def row6473 : SortedNatTree.Tree := (.node 13649 (.node 9833 (.node 8699 (.node 7643 (.node 6701 .empty .empty) .empty) (.node 9497 (.node 9221 .empty .empty) .empty)) (.node 11807 (.node 10883 (.node 10733 .empty .empty) .empty) (.node 13487 (.node 13421 .empty .empty) .empty))) (.node 16889 (.node 14639 (.node 13967 (.node 13691 .empty .empty) .empty) (.node 16619 (.node 15083 .empty .empty) .empty)) (.node 18593 (.node 18059 (.node 17093 .empty .empty) .empty) (.node 19433 (.node 19139 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6473_ordered : SortedNatTree.ordered row6473 = true := by decide
set_option maxRecDepth 4096 in
theorem row6473_values : SortedNatTree.values row6473 = Data.row6473 := rfl
theorem row6473_checked : SortedNatTree.contains row6473 = Data.row6473.contains := by
  rw [SortedNatTree.contains_eq_list row6473 row6473_ordered, row6473_values]
def row6481 : SortedNatTree.Tree := (.node 13789 (.node 10429 (.node 8287 (.node 7603 (.node 6793 .empty .empty) .empty) (.node 9463 .empty .empty)) (.node 12697 (.node 12541 (.node 11941 .empty .empty) .empty) (.node 13441 .empty .empty))) (.node 16831 (.node 16363 (.node 16111 (.node 14827 .empty .empty) .empty) (.node 16741 .empty .empty)) (.node 17377 (.node 17209 .empty .empty) (.node 17737 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6481_ordered : SortedNatTree.ordered row6481 = true := by decide
set_option maxRecDepth 4096 in
theorem row6481_values : SortedNatTree.values row6481 = Data.row6481 := rfl
theorem row6481_checked : SortedNatTree.contains row6481 = Data.row6481.contains := by
  rw [SortedNatTree.contains_eq_list row6481 row6481_ordered, row6481_values]
def row6491 : SortedNatTree.Tree := (.node 11597 (.node 9161 (.node 8237 (.node 7253 (.node 6983 (.node 6563 .empty .empty) .empty) (.node 7583 .empty .empty)) (.node 8663 (.node 8429 .empty .empty) (.node 8783 .empty .empty))) (.node 11159 (.node 10289 (.node 9791 (.node 9473 .empty .empty) .empty) (.node 10883 .empty .empty)) (.node 11447 (.node 11171 .empty .empty) (.node 11471 .empty .empty)))) (.node 15329 (.node 13259 (.node 12983 (.node 12227 (.node 11699 .empty .empty) .empty) (.node 13109 .empty .empty)) (.node 14657 (.node 13781 .empty .empty) (.node 14957 .empty .empty))) (.node 17333 (.node 17099 (.node 15569 (.node 15461 .empty .empty) .empty) (.node 17327 .empty .empty)) (.node 18839 (.node 17627 .empty .empty) (.node 19433 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6491_ordered : SortedNatTree.ordered row6491 = true := by decide
set_option maxRecDepth 4096 in
theorem row6491_values : SortedNatTree.values row6491 = Data.row6491 := rfl
theorem row6491_checked : SortedNatTree.contains row6491 = Data.row6491.contains := by
  rw [SortedNatTree.contains_eq_list row6491 row6491_ordered, row6491_values]
def row6521 : SortedNatTree.Tree := (.node 14249 (.node 10061 (.node 8039 (.node 7589 (.node 7043 .empty .empty) .empty) (.node 8663 (.node 8369 .empty .empty) .empty)) (.node 12743 (.node 11807 (.node 10181 .empty .empty) .empty) (.node 13841 .empty .empty))) (.node 16943 (.node 15227 (.node 14831 (.node 14327 .empty .empty) .empty) (.node 16691 (.node 16349 .empty .empty) .empty)) (.node 18371 (.node 18191 (.node 17189 .empty .empty) .empty) (.node 19079 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6521_ordered : SortedNatTree.ordered row6521 = true := by decide
set_option maxRecDepth 4096 in
theorem row6521_values : SortedNatTree.values row6521 = Data.row6521 := rfl
theorem row6521_checked : SortedNatTree.contains row6521 = Data.row6521.contains := by
  rw [SortedNatTree.contains_eq_list row6521 row6521_ordered, row6521_values]
def row6529 : SortedNatTree.Tree := (.node 11059 (.node 8329 (.node 7027 (.node 6961 (.node 6547 .empty .empty) .empty) (.node 8221 (.node 7549 .empty .empty) .empty)) (.node 8689 (.node 8647 (.node 8461 .empty .empty) .empty) (.node 10597 .empty .empty))) (.node 16651 (.node 14407 (.node 13399 (.node 11353 .empty .empty) .empty) (.node 15919 (.node 15643 .empty .empty) .empty)) (.node 18913 (.node 17401 (.node 17299 .empty .empty) .empty) (.node 19381 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6529_ordered : SortedNatTree.ordered row6529 = true := by decide
set_option maxRecDepth 4096 in
theorem row6529_values : SortedNatTree.values row6529 = Data.row6529 := rfl
theorem row6529_checked : SortedNatTree.contains row6529 = Data.row6529.contains := by
  rw [SortedNatTree.contains_eq_list row6529 row6529_ordered, row6529_values]
def row6547 : SortedNatTree.Tree := (.node 10453 (.node 7681 (.node 6871 (.node 6703 (.node 6637 .empty .empty) (.node 6841 .empty .empty)) (.node 7561 (.node 6967 .empty .empty) .empty)) (.node 9241 (.node 8689 (.node 7723 .empty .empty) (.node 8779 .empty .empty)) (.node 10267 (.node 10039 .empty .empty) .empty))) (.node 13921 (.node 12433 (.node 11677 (.node 11059 .empty .empty) (.node 11923 .empty .empty)) (.node 12907 (.node 12511 .empty .empty) .empty)) (.node 15331 (.node 14323 (.node 14107 .empty .empty) .empty) (.node 17929 (.node 15583 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6547_ordered : SortedNatTree.ordered row6547 = true := by decide
set_option maxRecDepth 4096 in
theorem row6547_values : SortedNatTree.values row6547 = Data.row6547 := rfl
theorem row6547_checked : SortedNatTree.contains row6547 = Data.row6547.contains := by
  rw [SortedNatTree.contains_eq_list row6547 row6547_ordered, row6547_values]
def row6551 : SortedNatTree.Tree := (.node 13229 (.node 9281 (.node 8093 (.node 7253 (.node 6653 .empty .empty) (.node 7481 .empty .empty)) (.node 8951 (.node 8543 .empty .empty) .empty)) (.node 12071 (.node 9539 (.node 9521 .empty .empty) (.node 11117 .empty .empty)) (.node 12569 (.node 12251 .empty .empty) .empty))) (.node 15971 (.node 15287 (.node 13829 (.node 13553 .empty .empty) (.node 14249 .empty .empty)) (.node 15767 (.node 15413 .empty .empty) .empty)) (.node 18311 (.node 18143 (.node 17189 .empty .empty) .empty) (.node 18773 (.node 18539 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6551_ordered : SortedNatTree.ordered row6551 = true := by decide
set_option maxRecDepth 4096 in
theorem row6551_values : SortedNatTree.values row6551 = Data.row6551 := rfl
theorem row6551_checked : SortedNatTree.contains row6551 = Data.row6551.contains := by
  rw [SortedNatTree.contains_eq_list row6551 row6551_ordered, row6551_values]
def row6553 : SortedNatTree.Tree := (.node 12517 (.node 10531 (.node 9631 (.node 8527 (.node 8329 .empty .empty) (.node 9421 .empty .empty)) (.node 10039 (.node 9907 .empty .empty) .empty)) (.node 11467 (.node 11131 (.node 10837 .empty .empty) (.node 11287 .empty .empty)) (.node 11617 (.node 11497 .empty .empty) .empty))) (.node 16477 (.node 15091 (.node 14449 (.node 13831 .empty .empty) (.node 14851 .empty .empty)) (.node 16339 (.node 15901 .empty .empty) .empty)) (.node 18127 (.node 17077 (.node 16747 .empty .empty) (.node 17293 .empty .empty)) (.node 19477 (.node 18787 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6553_ordered : SortedNatTree.ordered row6553 = true := by decide
set_option maxRecDepth 4096 in
theorem row6553_values : SortedNatTree.values row6553 = Data.row6553 := rfl
theorem row6553_checked : SortedNatTree.contains row6553 = Data.row6553.contains := by
  rw [SortedNatTree.contains_eq_list row6553 row6553_ordered, row6553_values]
def row6563 : SortedNatTree.Tree := (.node 14081 (.node 10427 (.node 8831 (.node 8537 (.node 6569 .empty .empty) (.node 8543 .empty .empty)) (.node 10139 (.node 9341 .empty .empty) .empty)) (.node 11597 (.node 11399 (.node 11273 .empty .empty) .empty) (.node 13241 (.node 12113 .empty .empty) .empty))) (.node 15629 (.node 15227 (.node 14957 (.node 14831 .empty .empty) (.node 14969 .empty .empty)) (.node 15467 (.node 15383 .empty .empty) .empty)) (.node 16217 (.node 16187 (.node 16007 .empty .empty) .empty) (.node 19457 (.node 19211 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6563_ordered : SortedNatTree.ordered row6563 = true := by decide
set_option maxRecDepth 4096 in
theorem row6563_values : SortedNatTree.values row6563 = Data.row6563 := rfl
theorem row6563_checked : SortedNatTree.contains row6563 = Data.row6563.contains := by
  rw [SortedNatTree.contains_eq_list row6563 row6563_ordered, row6563_values]
def row6569 : SortedNatTree.Tree := (.node 10607 (.node 8627 (.node 7109 (.node 6779 (.node 6689 .empty .empty) .empty) (.node 8537 .empty .empty)) (.node 9551 (.node 9209 (.node 9173 .empty .empty) .empty) (.node 10253 .empty .empty))) (.node 15647 (.node 12491 (.node 12239 (.node 12143 .empty .empty) .empty) (.node 13109 .empty .empty)) (.node 16529 (.node 16427 (.node 16361 .empty .empty) .empty) (.node 18959 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6569_ordered : SortedNatTree.ordered row6569 = true := by decide
set_option maxRecDepth 4096 in
theorem row6569_values : SortedNatTree.values row6569 = Data.row6569 := rfl
theorem row6569_checked : SortedNatTree.contains row6569 = Data.row6569.contains := by
  rw [SortedNatTree.contains_eq_list row6569 row6569_ordered, row6569_values]
def row6571 : SortedNatTree.Tree := (.node 12451 (.node 9103 (.node 7621 (.node 7129 (.node 6967 .empty .empty) (.node 7549 .empty .empty)) (.node 8293 (.node 8269 .empty .empty) (.node 8713 .empty .empty))) (.node 10939 (.node 9463 (.node 9439 .empty .empty) (.node 10429 .empty .empty)) (.node 12409 (.node 11593 .empty .empty) .empty))) (.node 14947 (.node 14149 (.node 13711 (.node 12889 .empty .empty) (.node 13729 .empty .empty)) (.node 14827 (.node 14173 .empty .empty) .empty)) (.node 15787 (.node 15373 (.node 15259 .empty .empty) (.node 15619 .empty .empty)) (.node 18229 (.node 17191 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6571_ordered : SortedNatTree.ordered row6571 = true := by decide
set_option maxRecDepth 4096 in
theorem row6571_values : SortedNatTree.values row6571 = Data.row6571 := rfl
theorem row6571_checked : SortedNatTree.contains row6571 = Data.row6571.contains := by
  rw [SortedNatTree.contains_eq_list row6571 row6571_ordered, row6571_values]
def row6577 : SortedNatTree.Tree := (.node 9787 (.node 7669 (.node 6883 (.node 6703 (.node 6661 .empty .empty) .empty) (.node 7603 (.node 7213 .empty .empty) .empty)) (.node 9067 (.node 8269 (.node 7687 .empty .empty) .empty) (.node 9739 .empty .empty))) (.node 15187 (.node 10987 (.node 10789 (.node 9829 .empty .empty) .empty) (.node 14149 (.node 11491 .empty .empty) .empty)) (.node 17707 (.node 15493 (.node 15391 .empty .empty) .empty) (.node 18223 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6577_ordered : SortedNatTree.ordered row6577 = true := by decide
set_option maxRecDepth 4096 in
theorem row6577_values : SortedNatTree.values row6577 = Data.row6577 := rfl
theorem row6577_checked : SortedNatTree.contains row6577 = Data.row6577.contains := by
  rw [SortedNatTree.contains_eq_list row6577 row6577_ordered, row6577_values]
def row6581 : SortedNatTree.Tree := (.node 13499 (.node 8171 (.node 7541 (.node 6857 .empty .empty) (.node 7883 .empty .empty)) (.node 12239 (.node 10301 .empty .empty) (.node 12473 .empty .empty))) (.node 17987 (.node 14207 (.node 13721 .empty .empty) (.node 15173 .empty .empty)) (.node 18077 (.node 18047 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row6581_ordered : SortedNatTree.ordered row6581 = true := by decide
set_option maxRecDepth 4096 in
theorem row6581_values : SortedNatTree.values row6581 = Data.row6581 := rfl
theorem row6581_checked : SortedNatTree.contains row6581 = Data.row6581.contains := by
  rw [SortedNatTree.contains_eq_list row6581 row6581_ordered, row6581_values]
def row6599 : SortedNatTree.Tree := (.node 12641 (.node 9767 (.node 8111 (.node 8081 (.node 7499 .empty .empty) .empty) (.node 9239 (.node 9161 .empty .empty) .empty)) (.node 10271 (.node 10247 (.node 9929 .empty .empty) .empty) (.node 12041 .empty .empty))) (.node 16889 (.node 15107 (.node 13043 (.node 12791 .empty .empty) .empty) (.node 15233 .empty .empty)) (.node 18269 (.node 18131 (.node 17789 .empty .empty) .empty) (.node 18719 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6599_ordered : SortedNatTree.ordered row6599 = true := by decide
set_option maxRecDepth 4096 in
theorem row6599_values : SortedNatTree.values row6599 = Data.row6599 := rfl
theorem row6599_checked : SortedNatTree.contains row6599 = Data.row6599.contains := by
  rw [SortedNatTree.contains_eq_list row6599 row6599_ordered, row6599_values]
def row6607 : SortedNatTree.Tree := (.node 11827 (.node 9199 (.node 8461 (.node 7867 (.node 6763 .empty .empty) (.node 8431 .empty .empty)) (.node 9157 (.node 9049 .empty .empty) .empty)) (.node 11071 (.node 10753 (.node 9439 .empty .empty) .empty) (.node 11497 (.node 11299 .empty .empty) .empty))) (.node 17137 (.node 15277 (.node 14251 (.node 12277 .empty .empty) .empty) (.node 16369 (.node 16183 .empty .empty) .empty)) (.node 18427 (.node 17839 (.node 17713 .empty .empty) .empty) (.node 19207 (.node 18637 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6607_ordered : SortedNatTree.ordered row6607 = true := by decide
set_option maxRecDepth 4096 in
theorem row6607_values : SortedNatTree.values row6607 = Data.row6607 := rfl
theorem row6607_checked : SortedNatTree.contains row6607 = Data.row6607.contains := by
  rw [SortedNatTree.contains_eq_list row6607 row6607_ordered, row6607_values]
def row6619 : SortedNatTree.Tree := (.node 10159 (.node 8689 (.node 8101 (.node 7603 (.node 7591 .empty .empty) .empty) (.node 8311 (.node 8179 .empty .empty) .empty)) (.node 9739 (.node 9319 (.node 9109 .empty .empty) .empty) (.node 10141 .empty .empty))) (.node 13633 (.node 10987 (.node 10867 (.node 10789 .empty .empty) .empty) (.node 12073 (.node 11743 .empty .empty) .empty)) (.node 16447 (.node 16111 (.node 15493 .empty .empty) .empty) (.node 18553 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6619_ordered : SortedNatTree.ordered row6619 = true := by decide
set_option maxRecDepth 4096 in
theorem row6619_values : SortedNatTree.values row6619 = Data.row6619 := rfl
theorem row6619_checked : SortedNatTree.contains row6619 = Data.row6619.contains := by
  rw [SortedNatTree.contains_eq_list row6619 row6619_ordered, row6619_values]
def row6637 : SortedNatTree.Tree := (.node 12043 (.node 9013 (.node 7351 (.node 7039 (.node 6997 (.node 6841 .empty .empty) .empty) (.node 7309 .empty .empty)) (.node 8527 (.node 8233 .empty .empty) (.node 8563 .empty .empty))) (.node 11047 (.node 10513 (.node 9337 (.node 9283 .empty .empty) .empty) (.node 10903 .empty .empty)) (.node 11677 (.node 11329 .empty .empty) (.node 12037 .empty .empty)))) (.node 15919 (.node 13513 (.node 12601 (.node 12457 (.node 12289 .empty .empty) .empty) (.node 13399 .empty .empty)) (.node 14347 (.node 14011 .empty .empty) (.node 15907 .empty .empty))) (.node 17713 (.node 16411 (.node 16381 .empty .empty) (.node 16927 .empty .empty)) (.node 18379 (.node 17749 .empty .empty) (.node 19183 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6637_ordered : SortedNatTree.ordered row6637 = true := by decide
set_option maxRecDepth 4096 in
theorem row6637_values : SortedNatTree.values row6637 = Data.row6637 := rfl
theorem row6637_checked : SortedNatTree.contains row6637 = Data.row6637.contains := by
  rw [SortedNatTree.contains_eq_list row6637 row6637_ordered, row6637_values]
def row6653 : SortedNatTree.Tree := (.node 12491 (.node 9203 (.node 8363 (.node 7649 (.node 7103 .empty .empty) .empty) (.node 9041 (.node 8819 .empty .empty) .empty)) (.node 10253 (.node 9923 (.node 9749 .empty .empty) .empty) (.node 10847 .empty .empty))) (.node 16187 (.node 14207 (.node 13487 (.node 13463 .empty .empty) .empty) (.node 15671 (.node 15473 .empty .empty) .empty)) (.node 16763 (.node 16649 (.node 16427 .empty .empty) .empty) (.node 18191 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6653_ordered : SortedNatTree.ordered row6653 = true := by decide
set_option maxRecDepth 4096 in
theorem row6653_values : SortedNatTree.values row6653 = Data.row6653 := rfl
theorem row6653_checked : SortedNatTree.contains row6653 = Data.row6653.contains := by
  rw [SortedNatTree.contains_eq_list row6653 row6653_ordered, row6653_values]
def row6659 : SortedNatTree.Tree := (.node 12809 (.node 9833 (.node 9623 (.node 8867 (.node 7043 .empty .empty) .empty) (.node 9791 .empty .empty)) (.node 11549 (.node 10799 (.node 10631 .empty .empty) .empty) (.node 12647 .empty .empty))) (.node 16073 (.node 15269 (.node 14723 (.node 14387 .empty .empty) .empty) (.node 15413 .empty .empty)) (.node 17117 (.node 16193 .empty .empty) (.node 18353 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6659_ordered : SortedNatTree.ordered row6659 = true := by decide
set_option maxRecDepth 4096 in
theorem row6659_values : SortedNatTree.values row6659 = Data.row6659 := rfl
theorem row6659_checked : SortedNatTree.contains row6659 = Data.row6659.contains := by
  rw [SortedNatTree.contains_eq_list row6659 row6659_ordered, row6659_values]
def row6661 : SortedNatTree.Tree := (.node 14389 (.node 12157 (.node 9901 (.node 8311 (.node 7687 .empty .empty) .empty) (.node 11779 (.node 11113 .empty .empty) .empty)) (.node 13729 (.node 13309 (.node 12829 .empty .empty) .empty) (.node 13759 .empty .empty))) (.node 15787 (.node 15313 (.node 15013 (.node 14563 .empty .empty) .empty) (.node 15733 .empty .empty)) (.node 18253 (.node 17299 (.node 17293 .empty .empty) .empty) (.node 18307 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6661_ordered : SortedNatTree.ordered row6661 = true := by decide
set_option maxRecDepth 4096 in
theorem row6661_values : SortedNatTree.values row6661 = Data.row6661 := rfl
theorem row6661_checked : SortedNatTree.contains row6661 = Data.row6661.contains := by
  rw [SortedNatTree.contains_eq_list row6661 row6661_ordered, row6661_values]
def row6673 : SortedNatTree.Tree := (.node 10771 (.node 8563 (.node 7639 (.node 7459 (.node 7351 .empty .empty) .empty) (.node 8293 .empty .empty)) (.node 9697 (.node 9661 (.node 8677 .empty .empty) .empty) (.node 10513 .empty .empty))) (.node 16987 (.node 15373 (.node 13063 (.node 12739 .empty .empty) .empty) (.node 15973 .empty .empty)) (.node 17851 (.node 17401 .empty .empty) (.node 18859 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6673_ordered : SortedNatTree.ordered row6673 = true := by decide
set_option maxRecDepth 4096 in
theorem row6673_values : SortedNatTree.values row6673 = Data.row6673 := rfl
theorem row6673_checked : SortedNatTree.contains row6673 = Data.row6673.contains := by
  rw [SortedNatTree.contains_eq_list row6673 row6673_ordered, row6673_values]
def row6679 : SortedNatTree.Tree := (.node 12979 (.node 9547 (.node 8353 (.node 8059 (.node 6967 .empty .empty) (.node 8161 .empty .empty)) (.node 9277 (.node 8863 .empty .empty) .empty)) (.node 11071 (.node 10597 (.node 9781 .empty .empty) .empty) (.node 12781 (.node 11941 .empty .empty) .empty))) (.node 16477 (.node 13933 (.node 13513 (.node 13003 .empty .empty) (.node 13633 .empty .empty)) (.node 16411 (.node 15583 .empty .empty) .empty)) (.node 18553 (.node 17923 (.node 17749 .empty .empty) .empty) (.node 19249 (.node 19207 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6679_ordered : SortedNatTree.ordered row6679 = true := by decide
set_option maxRecDepth 4096 in
theorem row6679_values : SortedNatTree.values row6679 = Data.row6679 := rfl
theorem row6679_checked : SortedNatTree.contains row6679 = Data.row6679.contains := by
  rw [SortedNatTree.contains_eq_list row6679 row6679_ordered, row6679_values]
def row6689 : SortedNatTree.Tree := (.node 12113 (.node 7949 (.node 7487 (.node 7151 (.node 6779 .empty .empty) (.node 7331 .empty .empty)) (.node 7673 (.node 7583 .empty .empty) .empty)) (.node 10973 (.node 8963 (.node 8537 .empty .empty) (.node 10937 .empty .empty)) (.node 12071 (.node 11657 .empty .empty) .empty))) (.node 16649 (.node 14423 (.node 12401 (.node 12269 .empty .empty) (.node 12659 .empty .empty)) (.node 16487 (.node 15461 .empty .empty) .empty)) (.node 17351 (.node 16943 (.node 16823 .empty .empty) (.node 17231 .empty .empty)) (.node 18251 (.node 17573 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6689_ordered : SortedNatTree.ordered row6689 = true := by decide
set_option maxRecDepth 4096 in
theorem row6689_values : SortedNatTree.values row6689 = Data.row6689 := rfl
theorem row6689_checked : SortedNatTree.contains row6689 = Data.row6689.contains := by
  rw [SortedNatTree.contains_eq_list row6689 row6689_ordered, row6689_values]
def row6691 : SortedNatTree.Tree := (.node 10723 (.node 8419 (.node 7687 (.node 7243 (.node 6781 .empty .empty) (.node 7507 .empty .empty)) (.node 8377 (.node 8329 .empty .empty) (.node 8389 .empty .empty))) (.node 9649 (.node 9343 (.node 9043 .empty .empty) (.node 9631 .empty .empty)) (.node 10009 (.node 9817 .empty .empty) (.node 10093 .empty .empty)))) (.node 15739 (.node 14197 (.node 12421 (.node 10861 .empty .empty) (.node 12487 .empty .empty)) (.node 15091 (.node 14533 .empty .empty) (.node 15679 .empty .empty))) (.node 17959 (.node 16561 (.node 16087 .empty .empty) (.node 17077 .empty .empty)) (.node 18661 (.node 18523 .empty .empty) (.node 19183 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row6691_ordered : SortedNatTree.ordered row6691 = true := by decide
set_option maxRecDepth 4096 in
theorem row6691_values : SortedNatTree.values row6691 = Data.row6691 := rfl
theorem row6691_checked : SortedNatTree.contains row6691 = Data.row6691.contains := by
  rw [SortedNatTree.contains_eq_list row6691 row6691_ordered, row6691_values]
def row6701 : SortedNatTree.Tree := (.node 11411 (.node 9629 (.node 8117 (.node 7019 (.node 6899 .empty .empty) (.node 8069 .empty .empty)) (.node 9431 (.node 8423 .empty .empty) (.node 9587 .empty .empty))) (.node 10781 (.node 10079 (.node 9857 .empty .empty) (.node 10733 .empty .empty)) (.node 11261 (.node 11213 .empty .empty) .empty))) (.node 16763 (.node 13721 (.node 12941 (.node 12689 .empty .empty) (.node 13043 .empty .empty)) (.node 15773 (.node 14009 .empty .empty) (.node 16193 .empty .empty))) (.node 18503 (.node 17597 (.node 17159 .empty .empty) (.node 18143 .empty .empty)) (.node 19301 (.node 18587 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6701_ordered : SortedNatTree.ordered row6701 = true := by decide
set_option maxRecDepth 4096 in
theorem row6701_values : SortedNatTree.values row6701 = Data.row6701 := rfl
theorem row6701_checked : SortedNatTree.contains row6701 = Data.row6701.contains := by
  rw [SortedNatTree.contains_eq_list row6701 row6701_ordered, row6701_values]
def row6703 : SortedNatTree.Tree := (.node 12211 (.node 8221 (.node 7459 (.node 7297 (.node 7213 .empty .empty) .empty) (.node 7723 (.node 7603 .empty .empty) .empty)) (.node 9187 (.node 9103 (.node 8431 .empty .empty) .empty) (.node 10141 (.node 9199 .empty .empty) .empty))) (.node 15817 (.node 14551 (.node 14341 (.node 14221 .empty .empty) .empty) (.node 15313 (.node 14731 .empty .empty) .empty)) (.node 16417 (.node 16249 (.node 15907 .empty .empty) .empty) (.node 18637 (.node 18097 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6703_ordered : SortedNatTree.ordered row6703 = true := by decide
set_option maxRecDepth 4096 in
theorem row6703_values : SortedNatTree.values row6703 = Data.row6703 := rfl
theorem row6703_checked : SortedNatTree.contains row6703 = Data.row6703.contains := by
  rw [SortedNatTree.contains_eq_list row6703 row6703_ordered, row6703_values]
def row6709 : SortedNatTree.Tree := (.node 13183 (.node 10861 (.node 8971 (.node 8839 (.node 7369 .empty .empty) .empty) (.node 10243 (.node 9103 .empty .empty) .empty)) (.node 12421 (.node 11113 (.node 10987 .empty .empty) .empty) (.node 12799 .empty .empty))) (.node 15559 (.node 13963 (.node 13669 (.node 13339 .empty .empty) .empty) (.node 14557 .empty .empty)) (.node 17791 (.node 16651 (.node 16249 .empty .empty) .empty) (.node 18517 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6709_ordered : SortedNatTree.ordered row6709 = true := by decide
set_option maxRecDepth 4096 in
theorem row6709_values : SortedNatTree.values row6709 = Data.row6709 := rfl
theorem row6709_checked : SortedNatTree.contains row6709 = Data.row6709.contains := by
  rw [SortedNatTree.contains_eq_list row6709 row6709_ordered, row6709_values]
def row6719 : SortedNatTree.Tree := (.node 11789 (.node 8237 (.node 7607 (.node 7523 (.node 7349 .empty .empty) (.node 7589 .empty .empty)) (.node 8147 (.node 7829 .empty .empty) .empty)) (.node 10487 (.node 9203 (.node 8693 .empty .empty) (.node 10427 .empty .empty)) (.node 11171 (.node 10853 .empty .empty) .empty))) (.node 16811 (.node 16253 (.node 13997 (.node 12809 .empty .empty) (.node 15887 .empty .empty)) (.node 16673 (.node 16493 .empty .empty) .empty)) (.node 18089 (.node 18059 (.node 17291 .empty .empty) .empty) (.node 18917 (.node 18539 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6719_ordered : SortedNatTree.ordered row6719 = true := by decide
set_option maxRecDepth 4096 in
theorem row6719_values : SortedNatTree.values row6719 = Data.row6719 := rfl
theorem row6719_checked : SortedNatTree.contains row6719 = Data.row6719.contains := by
  rw [SortedNatTree.contains_eq_list row6719 row6719_ordered, row6719_values]
def row6733 : SortedNatTree.Tree := (.node 11491 (.node 8839 (.node 8389 (.node 7789 (.node 7321 .empty .empty) (.node 7993 .empty .empty)) (.node 8599 (.node 8539 .empty .empty) .empty)) (.node 10321 (.node 9127 (.node 9043 .empty .empty) .empty) (.node 11239 (.node 10531 .empty .empty) .empty))) (.node 17209 (.node 13159 (.node 12433 (.node 12163 .empty .empty) .empty) (.node 16087 (.node 15733 .empty .empty) .empty)) (.node 18043 (.node 17707 (.node 17431 .empty .empty) .empty) (.node 19237 (.node 18379 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6733_ordered : SortedNatTree.ordered row6733 = true := by decide
set_option maxRecDepth 4096 in
theorem row6733_values : SortedNatTree.values row6733 = Data.row6733 := rfl
theorem row6733_checked : SortedNatTree.contains row6733 = Data.row6733.contains := by
  rw [SortedNatTree.contains_eq_list row6733 row6733_ordered, row6733_values]
def row6737 : SortedNatTree.Tree := (.node 10979 (.node 10091 (.node 8627 (.node 7451 (.node 6779 .empty .empty) .empty) (.node 9677 (.node 9161 .empty .empty) .empty)) (.node 10601 (.node 10589 (.node 10463 .empty .empty) .empty) (.node 10853 .empty .empty))) (.node 13709 (.node 13163 (.node 12101 (.node 11681 .empty .empty) .empty) (.node 13619 .empty .empty)) (.node 16481 (.node 15017 (.node 14543 .empty .empty) .empty) (.node 18911 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6737_ordered : SortedNatTree.ordered row6737 = true := by decide
set_option maxRecDepth 4096 in
theorem row6737_values : SortedNatTree.values row6737 = Data.row6737 := rfl
theorem row6737_checked : SortedNatTree.contains row6737 = Data.row6737.contains := by
  rw [SortedNatTree.contains_eq_list row6737 row6737_ordered, row6737_values]
def row6761 : SortedNatTree.Tree := (.node 14951 (.node 9749 (.node 8669 (.node 8387 (.node 7229 .empty .empty) .empty) (.node 9311 (.node 8867 .empty .empty) .empty)) (.node 11633 (.node 10499 (.node 10151 .empty .empty) .empty) (.node 14081 .empty .empty))) (.node 17207 (.node 16607 (.node 16451 (.node 16127 .empty .empty) .empty) (.node 17093 .empty .empty)) (.node 18329 (.node 17891 (.node 17597 .empty .empty) .empty) (.node 18617 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6761_ordered : SortedNatTree.ordered row6761 = true := by decide
set_option maxRecDepth 4096 in
theorem row6761_values : SortedNatTree.values row6761 = Data.row6761 := rfl
theorem row6761_checked : SortedNatTree.contains row6761 = Data.row6761.contains := by
  rw [SortedNatTree.contains_eq_list row6761 row6761_ordered, row6761_values]
def row6763 : SortedNatTree.Tree := (.node 13633 (.node 9883 (.node 8863 (.node 8287 (.node 7333 .empty .empty) (.node 8581 .empty .empty)) (.node 9631 (.node 9001 .empty .empty) (.node 9769 .empty .empty))) (.node 12433 (.node 11887 (.node 11677 .empty .empty) (.node 12379 .empty .empty)) (.node 13219 (.node 12919 .empty .empty) .empty))) (.node 16519 (.node 15973 (.node 14407 (.node 14347 .empty .empty) (.node 15427 .empty .empty)) (.node 16141 (.node 16111 .empty .empty) .empty)) (.node 17959 (.node 17167 (.node 16573 .empty .empty) (.node 17791 .empty .empty)) (.node 18553 (.node 17989 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6763_ordered : SortedNatTree.ordered row6763 = true := by decide
set_option maxRecDepth 4096 in
theorem row6763_values : SortedNatTree.values row6763 = Data.row6763 := rfl
theorem row6763_checked : SortedNatTree.contains row6763 = Data.row6763.contains := by
  rw [SortedNatTree.contains_eq_list row6763 row6763_ordered, row6763_values]
def row6779 : SortedNatTree.Tree := (.node 9749 (.node 8609 (.node 8231 (.node 7517 (.node 7487 .empty .empty) (.node 7823 .empty .empty)) (.node 8537 (.node 8369 .empty .empty) .empty)) (.node 9497 (.node 9257 (.node 8663 .empty .empty) (.node 9491 .empty .empty)) (.node 9551 (.node 9539 .empty .empty) .empty))) (.node 16319 (.node 11783 (.node 10859 (.node 10691 .empty .empty) (.node 11489 .empty .empty)) (.node 12659 (.node 12401 .empty .empty) .empty)) (.node 17351 (.node 16931 (.node 16823 .empty .empty) .empty) (.node 19001 (.node 18617 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6779_ordered : SortedNatTree.ordered row6779 = true := by decide
set_option maxRecDepth 4096 in
theorem row6779_values : SortedNatTree.values row6779 = Data.row6779 := rfl
theorem row6779_checked : SortedNatTree.contains row6779 = Data.row6779.contains := by
  rw [SortedNatTree.contains_eq_list row6779 row6779_ordered, row6779_values]
def row6781 : SortedNatTree.Tree := (.node 13441 (.node 11353 (.node 7237 (.node 6841 (.node 6823 .empty .empty) (.node 7069 .empty .empty)) (.node 9439 (.node 8161 .empty .empty) .empty)) (.node 12277 (.node 11971 (.node 11923 .empty .empty) .empty) (.node 13417 (.node 12973 .empty .empty) .empty))) (.node 15583 (.node 14797 (.node 13831 (.node 13729 .empty .empty) .empty) (.node 15259 (.node 14929 .empty .empty) .empty)) (.node 18787 (.node 17443 (.node 15679 .empty .empty) .empty) (.node 19231 (.node 19087 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6781_ordered : SortedNatTree.ordered row6781 = true := by decide
set_option maxRecDepth 4096 in
theorem row6781_values : SortedNatTree.values row6781 = Data.row6781 := rfl
theorem row6781_checked : SortedNatTree.contains row6781 = Data.row6781.contains := by
  rw [SortedNatTree.contains_eq_list row6781 row6781_ordered, row6781_values]
def row6791 : SortedNatTree.Tree := (.node 11621 (.node 9323 (.node 8831 (.node 7841 (.node 6899 .empty .empty) .empty) (.node 9227 .empty .empty)) (.node 10007 (.node 9377 .empty .empty) (.node 11483 .empty .empty))) (.node 15887 (.node 15299 (.node 14627 (.node 12281 .empty .empty) .empty) (.node 15467 .empty .empty)) (.node 18959 (.node 16703 .empty .empty) (.node 19079 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6791_ordered : SortedNatTree.ordered row6791 = true := by decide
set_option maxRecDepth 4096 in
theorem row6791_values : SortedNatTree.values row6791 = Data.row6791 := rfl
theorem row6791_checked : SortedNatTree.contains row6791 = Data.row6791.contains := by
  rw [SortedNatTree.contains_eq_list row6791 row6791_ordered, row6791_values]
def row6793 : SortedNatTree.Tree := (.node 10369 (.node 8467 (.node 7717 (.node 7159 (.node 7039 .empty .empty) .empty) (.node 8293 (.node 7951 .empty .empty) .empty)) (.node 9127 (.node 9049 (.node 8707 .empty .empty) .empty) (.node 9901 (.node 9433 .empty .empty) .empty))) (.node 15451 (.node 11953 (.node 11833 (.node 10663 .empty .empty) .empty) (.node 15349 (.node 12619 .empty .empty) .empty)) (.node 15739 (.node 15727 (.node 15601 .empty .empty) .empty) (.node 18919 (.node 18253 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6793_ordered : SortedNatTree.ordered row6793 = true := by decide
set_option maxRecDepth 4096 in
theorem row6793_values : SortedNatTree.values row6793 = Data.row6793 := rfl
theorem row6793_checked : SortedNatTree.contains row6793 = Data.row6793.contains := by
  rw [SortedNatTree.contains_eq_list row6793 row6793_ordered, row6793_values]
def row6803 : SortedNatTree.Tree := (.node 13553 (.node 9749 (.node 9281 (.node 7649 (.node 6977 .empty .empty) (.node 8123 .empty .empty)) (.node 9743 (.node 9533 .empty .empty) .empty)) (.node 10937 (.node 10133 (.node 9791 .empty .empty) (.node 10889 .empty .empty)) (.node 13313 (.node 13229 .empty .empty) .empty))) (.node 16937 (.node 14879 (.node 14549 (.node 13577 .empty .empty) (.node 14753 .empty .empty)) (.node 15803 (.node 15641 .empty .empty) .empty)) (.node 18539 (.node 17681 (.node 17321 .empty .empty) .empty) (.node 19163 (.node 18719 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6803_ordered : SortedNatTree.ordered row6803 = true := by decide
set_option maxRecDepth 4096 in
theorem row6803_values : SortedNatTree.values row6803 = Data.row6803 := rfl
theorem row6803_checked : SortedNatTree.contains row6803 = Data.row6803.contains := by
  rw [SortedNatTree.contains_eq_list row6803 row6803_ordered, row6803_values]
def row6823 : SortedNatTree.Tree := (.node 10177 (.node 9109 (.node 8059 (.node 7951 (.node 7867 .empty .empty) (.node 7993 .empty .empty)) (.node 8641 (.node 8311 .empty .empty) (.node 9043 .empty .empty))) (.node 9967 (.node 9319 (.node 9133 .empty .empty) (.node 9511 .empty .empty)) (.node 10141 (.node 10099 .empty .empty) .empty))) (.node 13873 (.node 12703 (.node 12409 (.node 11131 .empty .empty) (.node 12637 .empty .empty)) (.node 13789 (.node 13183 .empty .empty) .empty)) (.node 18199 (.node 14419 (.node 14107 .empty .empty) (.node 15031 .empty .empty)) (.node 18691 (.node 18451 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6823_ordered : SortedNatTree.ordered row6823 = true := by decide
set_option maxRecDepth 4096 in
theorem row6823_values : SortedNatTree.values row6823 = Data.row6823 := rfl
theorem row6823_checked : SortedNatTree.contains row6823 = Data.row6823.contains := by
  rw [SortedNatTree.contains_eq_list row6823 row6823_ordered, row6823_values]
def row6827 : SortedNatTree.Tree := (.node 10781 (.node 8699 (.node 7499 (.node 7103 (.node 6977 .empty .empty) (.node 7487 .empty .empty)) (.node 8369 (.node 7817 .empty .empty) .empty)) (.node 9587 (.node 9041 (.node 8807 .empty .empty) (.node 9413 .empty .empty)) (.node 9791 (.node 9623 .empty .empty) .empty))) (.node 16217 (.node 13103 (.node 12791 (.node 11423 .empty .empty) (.node 13043 .empty .empty)) (.node 13553 (.node 13469 .empty .empty) .empty)) (.node 18251 (.node 16673 (.node 16487 .empty .empty) .empty) (.node 19013 (.node 18401 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6827_ordered : SortedNatTree.ordered row6827 = true := by decide
set_option maxRecDepth 4096 in
theorem row6827_values : SortedNatTree.values row6827 = Data.row6827 := rfl
theorem row6827_checked : SortedNatTree.contains row6827 = Data.row6827.contains := by
  rw [SortedNatTree.contains_eq_list row6827 row6827_ordered, row6827_values]
def row6829 : SortedNatTree.Tree := (.node 12073 (.node 9697 (.node 7219 (.node 7213 (.node 7057 .empty .empty) .empty) (.node 8317 (.node 7507 .empty .empty) .empty)) (.node 10957 (.node 9769 (.node 9739 .empty .empty) .empty) (.node 11491 .empty .empty))) (.node 16987 (.node 15643 (.node 15139 (.node 13687 .empty .empty) .empty) (.node 16567 (.node 15907 .empty .empty) .empty)) (.node 17581 (.node 17419 (.node 17077 .empty .empty) .empty) (.node 17749 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6829_ordered : SortedNatTree.ordered row6829 = true := by decide
set_option maxRecDepth 4096 in
theorem row6829_values : SortedNatTree.values row6829 = Data.row6829 := rfl
theorem row6829_checked : SortedNatTree.contains row6829 = Data.row6829.contains := by
  rw [SortedNatTree.contains_eq_list row6829 row6829_ordered, row6829_values]
def row6833 : SortedNatTree.Tree := (.node 11399 (.node 8087 (.node 7487 (.node 6899 (.node 6869 .empty .empty) .empty) (.node 7673 .empty .empty)) (.node 10781 (.node 9521 (.node 8609 .empty .empty) .empty) (.node 10853 .empty .empty))) (.node 15329 (.node 13679 (.node 13313 (.node 12503 .empty .empty) .empty) (.node 14627 .empty .empty)) (.node 17093 (.node 15443 (.node 15401 .empty .empty) .empty) (.node 18803 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6833_ordered : SortedNatTree.ordered row6833 = true := by decide
set_option maxRecDepth 4096 in
theorem row6833_values : SortedNatTree.values row6833 = Data.row6833 := rfl
theorem row6833_checked : SortedNatTree.contains row6833 = Data.row6833.contains := by
  rw [SortedNatTree.contains_eq_list row6833 row6833_ordered, row6833_values]
def row6841 : SortedNatTree.Tree := (.node 13063 (.node 10663 (.node 7039 (.node 6991 .empty .empty) (.node 8377 .empty .empty)) (.node 12823 (.node 11383 .empty .empty) (.node 12889 .empty .empty))) (.node 15451 (.node 13627 (.node 13159 .empty .empty) (.node 14401 .empty .empty)) (.node 16921 (.node 15559 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row6841_ordered : SortedNatTree.ordered row6841 = true := by decide
set_option maxRecDepth 4096 in
theorem row6841_values : SortedNatTree.values row6841 = Data.row6841 := rfl
theorem row6841_checked : SortedNatTree.contains row6841 = Data.row6841.contains := by
  rw [SortedNatTree.contains_eq_list row6841 row6841_ordered, row6841_values]
def row6857 : SortedNatTree.Tree := (.node 14369 (.node 7907 (.node 7193 (.node 7151 (.node 7121 .empty .empty) .empty) (.node 7829 .empty .empty)) (.node 12479 (.node 12197 (.node 9473 .empty .empty) .empty) (.node 12821 .empty .empty))) (.node 16649 (.node 16007 (.node 15809 (.node 15227 .empty .empty) .empty) (.node 16091 .empty .empty)) (.node 17231 (.node 16943 .empty .empty) (.node 19031 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6857_ordered : SortedNatTree.ordered row6857 = true := by decide
set_option maxRecDepth 4096 in
theorem row6857_values : SortedNatTree.values row6857 = Data.row6857 := rfl
theorem row6857_checked : SortedNatTree.contains row6857 = Data.row6857.contains := by
  rw [SortedNatTree.contains_eq_list row6857 row6857_ordered, row6857_values]
def row6863 : SortedNatTree.Tree := (.node 13151 (.node 10181 (.node 8747 (.node 8501 (.node 8243 .empty .empty) (.node 8699 .empty .empty)) (.node 9587 (.node 8951 .empty .empty) .empty)) (.node 12071 (.node 11447 (.node 10463 .empty .empty) .empty) (.node 12527 (.node 12401 .empty .empty) .empty))) (.node 17231 (.node 16229 (.node 16217 (.node 14207 .empty .empty) .empty) (.node 17189 (.node 16901 .empty .empty) .empty)) (.node 17987 (.node 17747 (.node 17483 .empty .empty) .empty) (.node 18839 (.node 18671 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6863_ordered : SortedNatTree.ordered row6863 = true := by decide
set_option maxRecDepth 4096 in
theorem row6863_values : SortedNatTree.values row6863 = Data.row6863 := rfl
theorem row6863_checked : SortedNatTree.contains row6863 = Data.row6863.contains := by
  rw [SortedNatTree.contains_eq_list row6863 row6863_ordered, row6863_values]
def row6869 : SortedNatTree.Tree := (.node 12899 (.node 9623 (.node 7907 (.node 7121 (.node 6959 .empty .empty) (.node 7583 .empty .empty)) (.node 8111 (.node 8087 .empty .empty) (.node 8423 .empty .empty))) (.node 11243 (.node 10949 (.node 10607 .empty .empty) (.node 11117 .empty .empty)) (.node 11783 (.node 11321 .empty .empty) .empty))) (.node 15161 (.node 13907 (.node 13649 (.node 13259 .empty .empty) (.node 13697 .empty .empty)) (.node 15017 (.node 14051 .empty .empty) .empty)) (.node 17183 (.node 16193 (.node 15233 .empty .empty) (.node 17093 .empty .empty)) (.node 18839 (.node 18047 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6869_ordered : SortedNatTree.ordered row6869 = true := by decide
set_option maxRecDepth 4096 in
theorem row6869_values : SortedNatTree.values row6869 = Data.row6869 := rfl
theorem row6869_checked : SortedNatTree.contains row6869 = Data.row6869.contains := by
  rw [SortedNatTree.contains_eq_list row6869 row6869_ordered, row6869_values]
def row6871 : SortedNatTree.Tree := (.node 12763 (.node 10903 (.node 9697 (.node 8191 (.node 7039 .empty .empty) .empty) (.node 10477 (.node 10453 .empty .empty) .empty)) (.node 11833 (.node 11701 (.node 11329 .empty .empty) .empty) (.node 12619 .empty .empty))) (.node 15727 (.node 15391 (.node 13723 (.node 13597 .empty .empty) .empty) (.node 15661 (.node 15439 .empty .empty) .empty)) (.node 18013 (.node 16747 (.node 15739 .empty .empty) .empty) (.node 18301 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6871_ordered : SortedNatTree.ordered row6871 = true := by decide
set_option maxRecDepth 4096 in
theorem row6871_values : SortedNatTree.values row6871 = Data.row6871 := rfl
theorem row6871_checked : SortedNatTree.contains row6871 = Data.row6871.contains := by
  rw [SortedNatTree.contains_eq_list row6871 row6871_ordered, row6871_values]
def row6883 : SortedNatTree.Tree := (.node 13267 (.node 9973 (.node 8647 (.node 8293 (.node 7477 .empty .empty) .empty) (.node 9091 (.node 8719 .empty .empty) .empty)) (.node 12451 (.node 11353 (.node 10939 .empty .empty) .empty) (.node 13147 .empty .empty))) (.node 14947 (.node 14251 (.node 13669 (.node 13567 .empty .empty) .empty) (.node 14713 .empty .empty)) (.node 17299 (.node 16879 (.node 16267 .empty .empty) .empty) (.node 18859 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6883_ordered : SortedNatTree.ordered row6883 = true := by decide
set_option maxRecDepth 4096 in
theorem row6883_values : SortedNatTree.values row6883 = Data.row6883 := rfl
theorem row6883_checked : SortedNatTree.contains row6883 = Data.row6883.contains := by
  rw [SortedNatTree.contains_eq_list row6883 row6883_ordered, row6883_values]
def row6899 : SortedNatTree.Tree := (.node 12647 (.node 10391 (.node 8669 (.node 7127 (.node 6911 .empty .empty) (.node 7529 .empty .empty)) (.node 9551 (.node 8849 .empty .empty) (.node 10343 .empty .empty))) (.node 11447 (.node 10883 (.node 10433 .empty .empty) (.node 11093 .empty .empty)) (.node 12569 (.node 11597 .empty .empty) .empty))) (.node 14957 (.node 13967 (.node 13151 (.node 12911 .empty .empty) (.node 13613 .empty .empty)) (.node 14549 (.node 14051 .empty .empty) .empty)) (.node 17909 (.node 16229 (.node 16187 .empty .empty) (.node 17321 .empty .empty)) (.node 18413 (.node 18371 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6899_ordered : SortedNatTree.ordered row6899 = true := by decide
set_option maxRecDepth 4096 in
theorem row6899_values : SortedNatTree.values row6899 = Data.row6899 := rfl
theorem row6899_checked : SortedNatTree.contains row6899 = Data.row6899.contains := by
  rw [SortedNatTree.contains_eq_list row6899 row6899_ordered, row6899_values]
def row6907 : SortedNatTree.Tree := (.node 12973 (.node 9679 (.node 8689 (.node 7759 (.node 7549 .empty .empty) (.node 8431 .empty .empty)) (.node 9151 (.node 8839 .empty .empty) .empty)) (.node 11083 (.node 10729 (.node 10369 .empty .empty) (.node 10753 .empty .empty)) (.node 12697 (.node 12511 .empty .empty) .empty))) (.node 15427 (.node 13999 (.node 13789 (.node 13033 .empty .empty) (.node 13831 .empty .empty)) (.node 15217 (.node 14107 .empty .empty) .empty)) (.node 17971 (.node 17047 (.node 15937 .empty .empty) .empty) (.node 18367 (.node 17977 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6907_ordered : SortedNatTree.ordered row6907 = true := by decide
set_option maxRecDepth 4096 in
theorem row6907_values : SortedNatTree.values row6907 = Data.row6907 := rfl
theorem row6907_checked : SortedNatTree.contains row6907 = Data.row6907.contains := by
  rw [SortedNatTree.contains_eq_list row6907 row6907_ordered, row6907_values]
def row6911 : SortedNatTree.Tree := (.node 11321 (.node 9689 (.node 9227 (.node 9203 (.node 8969 .empty .empty) .empty) (.node 9629 .empty .empty)) (.node 10631 (.node 10499 (.node 9851 .empty .empty) .empty) (.node 10949 .empty .empty))) (.node 14951 (.node 13187 (.node 13001 (.node 11969 .empty .empty) .empty) (.node 13697 .empty .empty)) (.node 17159 (.node 15083 .empty .empty) (.node 18269 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6911_ordered : SortedNatTree.ordered row6911 = true := by decide
set_option maxRecDepth 4096 in
theorem row6911_values : SortedNatTree.values row6911 = Data.row6911 := rfl
theorem row6911_checked : SortedNatTree.contains row6911 = Data.row6911.contains := by
  rw [SortedNatTree.contains_eq_list row6911 row6911_ordered, row6911_values]
def row6917 : SortedNatTree.Tree := (.node 15053 (.node 10973 (.node 7331 (.node 7211 .empty .empty) (.node 8147 .empty .empty)) (.node 13457 (.node 11243 .empty .empty) (.node 13469 .empty .empty))) (.node 18089 (.node 16673 (.node 15527 .empty .empty) (.node 17093 .empty .empty)) (.node 18917 (.node 18869 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row6917_ordered : SortedNatTree.ordered row6917 = true := by decide
set_option maxRecDepth 4096 in
theorem row6917_values : SortedNatTree.values row6917 = Data.row6917 := rfl
theorem row6917_checked : SortedNatTree.contains row6917 = Data.row6917.contains := by
  rw [SortedNatTree.contains_eq_list row6917 row6917_ordered, row6917_values]
def row6947 : SortedNatTree.Tree := (.node 12269 (.node 8543 (.node 8081 (.node 7829 (.node 7013 .empty .empty) (.node 8069 .empty .empty)) (.node 8447 (.node 8369 .empty .empty) .empty)) (.node 11393 (.node 9533 (.node 9161 .empty .empty) (.node 9791 .empty .empty)) (.node 11657 (.node 11597 .empty .empty) .empty))) (.node 15881 (.node 14717 (.node 14051 (.node 13421 .empty .empty) (.node 14243 .empty .empty)) (.node 15773 (.node 15227 .empty .empty) .empty)) (.node 18119 (.node 17579 (.node 15887 .empty .empty) (.node 17909 .empty .empty)) (.node 19073 (.node 19037 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6947_ordered : SortedNatTree.ordered row6947 = true := by decide
set_option maxRecDepth 4096 in
theorem row6947_values : SortedNatTree.values row6947 = Data.row6947 := rfl
theorem row6947_checked : SortedNatTree.contains row6947 = Data.row6947.contains := by
  rw [SortedNatTree.contains_eq_list row6947 row6947_ordered, row6947_values]
def row6949 : SortedNatTree.Tree := (.node 11551 (.node 8419 (.node 7621 (.node 7591 (.node 6967 .empty .empty) .empty) (.node 7741 (.node 7723 .empty .empty) .empty)) (.node 10639 (.node 10159 (.node 8521 .empty .empty) .empty) (.node 11317 .empty .empty))) (.node 16333 (.node 14401 (.node 14251 (.node 12277 .empty .empty) .empty) (.node 15427 (.node 14713 .empty .empty) .empty)) (.node 18199 (.node 17659 (.node 16561 .empty .empty) .empty) (.node 18517 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6949_ordered : SortedNatTree.ordered row6949 = true := by decide
set_option maxRecDepth 4096 in
theorem row6949_values : SortedNatTree.values row6949 = Data.row6949 := rfl
theorem row6949_checked : SortedNatTree.contains row6949 = Data.row6949.contains := by
  rw [SortedNatTree.contains_eq_list row6949 row6949_ordered, row6949_values]
def row6959 : SortedNatTree.Tree := (.node 11171 (.node 8849 (.node 8093 (.node 7937 (.node 7193 .empty .empty) .empty) (.node 8747 (.node 8429 .empty .empty) .empty)) (.node 10331 (.node 9293 (.node 8861 .empty .empty) .empty) (.node 10499 .empty .empty))) (.node 15287 (.node 12911 (.node 11717 (.node 11273 .empty .empty) .empty) (.node 14489 (.node 12923 .empty .empty) .empty)) (.node 16829 (.node 16349 (.node 15971 .empty .empty) .empty) (.node 17291 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6959_ordered : SortedNatTree.ordered row6959 = true := by decide
set_option maxRecDepth 4096 in
theorem row6959_values : SortedNatTree.values row6959 = Data.row6959 := rfl
theorem row6959_checked : SortedNatTree.contains row6959 = Data.row6959.contains := by
  rw [SortedNatTree.contains_eq_list row6959 row6959_ordered, row6959_values]
def row6961 : SortedNatTree.Tree := (.node 11587 (.node 8623 (.node 8377 (.node 7243 (.node 7069 .empty .empty) (.node 7879 .empty .empty)) (.node 8581 (.node 8389 .empty .empty) .empty)) (.node 9421 (.node 9181 (.node 8803 .empty .empty) (.node 9391 .empty .empty)) (.node 10993 (.node 9817 .empty .empty) .empty))) (.node 16921 (.node 15313 (.node 13267 (.node 12721 .empty .empty) (.node 15187 .empty .empty)) (.node 15733 (.node 15319 .empty .empty) .empty)) (.node 17959 (.node 17791 (.node 17659 .empty .empty) .empty) (.node 18301 (.node 18043 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6961_ordered : SortedNatTree.ordered row6961 = true := by decide
set_option maxRecDepth 4096 in
theorem row6961_values : SortedNatTree.values row6961 = Data.row6961 := rfl
theorem row6961_checked : SortedNatTree.contains row6961 = Data.row6961.contains := by
  rw [SortedNatTree.contains_eq_list row6961 row6961_ordered, row6961_values]
def row6967 : SortedNatTree.Tree := (.node 11941 (.node 9781 (.node 8929 (.node 7687 (.node 7177 .empty .empty) (.node 8839 .empty .empty)) (.node 9679 (.node 9619 .empty .empty) .empty)) (.node 11383 (.node 10177 (.node 10159 .empty .empty) (.node 10429 .empty .empty)) (.node 11923 (.node 11593 .empty .empty) .empty))) (.node 15919 (.node 13291 (.node 12391 (.node 12157 .empty .empty) (.node 12637 .empty .empty)) (.node 14737 (.node 14419 .empty .empty) .empty)) (.node 17599 (.node 16249 (.node 16063 .empty .empty) .empty) (.node 18307 (.node 17911 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6967_ordered : SortedNatTree.ordered row6967 = true := by decide
set_option maxRecDepth 4096 in
theorem row6967_values : SortedNatTree.values row6967 = Data.row6967 := rfl
theorem row6967_checked : SortedNatTree.contains row6967 = Data.row6967.contains := by
  rw [SortedNatTree.contains_eq_list row6967 row6967_ordered, row6967_values]
def row6971 : SortedNatTree.Tree := (.node 13121 (.node 10781 (.node 9029 (.node 7673 (.node 7211 .empty .empty) .empty) (.node 9719 (.node 9059 .empty .empty) .empty)) (.node 11171 (.node 11087 (.node 10889 .empty .empty) .empty) (.node 12491 .empty .empty))) (.node 16649 (.node 14969 (.node 14939 (.node 13619 .empty .empty) .empty) (.node 15467 .empty .empty)) (.node 19001 (.node 17909 (.node 17903 .empty .empty) .empty) (.node 19037 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6971_ordered : SortedNatTree.ordered row6971 = true := by decide
set_option maxRecDepth 4096 in
theorem row6971_values : SortedNatTree.values row6971 = Data.row6971 := rfl
theorem row6971_checked : SortedNatTree.contains row6971 = Data.row6971.contains := by
  rw [SortedNatTree.contains_eq_list row6971 row6971_ordered, row6971_values]
def row6977 : SortedNatTree.Tree := (.node 12143 (.node 9833 (.node 9467 (.node 7577 (.node 7349 .empty .empty) (.node 8039 .empty .empty)) (.node 9767 (.node 9551 .empty .empty) .empty)) (.node 11447 (.node 11243 (.node 11171 .empty .empty) .empty) (.node 11867 (.node 11579 .empty .empty) .empty))) (.node 15527 (.node 14009 (.node 12941 (.node 12413 .empty .empty) (.node 13229 .empty .empty)) (.node 14411 (.node 14057 .empty .empty) .empty)) (.node 16883 (.node 16823 (.node 15731 .empty .empty) .empty) (.node 17747 (.node 16889 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row6977_ordered : SortedNatTree.ordered row6977 = true := by decide
set_option maxRecDepth 4096 in
theorem row6977_values : SortedNatTree.values row6977 = Data.row6977 := rfl
theorem row6977_checked : SortedNatTree.contains row6977 = Data.row6977.contains := by
  rw [SortedNatTree.contains_eq_list row6977 row6977_ordered, row6977_values]
def row6983 : SortedNatTree.Tree := (.node 12743 (.node 9677 (.node 8423 (.node 8117 (.node 7043 .empty .empty) .empty) (.node 9059 (.node 8969 .empty .empty) .empty)) (.node 10883 (.node 10559 (.node 10181 .empty .empty) .empty) (.node 12653 (.node 12569 .empty .empty) .empty))) (.node 16301 (.node 15263 (.node 14549 (.node 14537 .empty .empty) .empty) (.node 16097 (.node 15923 .empty .empty) .empty)) (.node 17957 (.node 17351 (.node 17231 .empty .empty) .empty) (.node 18743 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row6983_ordered : SortedNatTree.ordered row6983 = true := by decide
set_option maxRecDepth 4096 in
theorem row6983_values : SortedNatTree.values row6983 = Data.row6983 := rfl
theorem row6983_checked : SortedNatTree.contains row6983 = Data.row6983.contains := by
  rw [SortedNatTree.contains_eq_list row6983 row6983_ordered, row6983_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
