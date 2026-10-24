import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row5347 : SortedNatTree.Tree := (.node 13093 (.node 8539 (.node 8221 (.node 7027 (.node 5647 .empty .empty) (.node 7789 .empty .empty)) (.node 8419 (.node 8287 .empty .empty) (.node 8461 .empty .empty))) (.node 12547 (.node 11119 (.node 9007 .empty .empty) (.node 11701 .empty .empty)) (.node 13009 (.node 12967 .empty .empty) .empty))) (.node 17431 (.node 16141 (.node 15991 (.node 15607 .empty .empty) (.node 16111 .empty .empty)) (.node 17011 (.node 16381 .empty .empty) .empty)) (.node 18913 (.node 18199 (.node 17707 .empty .empty) (.node 18661 .empty .empty)) (.node 19813 (.node 19753 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5347_ordered : SortedNatTree.ordered row5347 = true := by decide
set_option maxRecDepth 4096 in
theorem row5347_values : SortedNatTree.values row5347 = Data.row5347 := rfl
theorem row5347_checked : SortedNatTree.contains row5347 = Data.row5347.contains := by
  rw [SortedNatTree.contains_eq_list row5347 row5347_ordered, row5347_values]
def row5351 : SortedNatTree.Tree := (.node 12323 (.node 8291 (.node 6131 (.node 5927 (.node 5879 (.node 5519 .empty .empty) .empty) (.node 6011 .empty .empty)) (.node 7019 (.node 6899 (.node 6761 .empty .empty) .empty) (.node 8039 .empty .empty))) (.node 9539 (.node 8747 (.node 8627 (.node 8537 .empty .empty) .empty) (.node 8951 .empty .empty)) (.node 11321 (.node 9833 .empty .empty) (.node 12203 .empty .empty)))) (.node 15683 (.node 13451 (.node 12911 (.node 12497 (.node 12401 .empty .empty) .empty) (.node 13043 .empty .empty)) (.node 15101 (.node 14303 .empty .empty) (.node 15299 .empty .empty))) (.node 17027 (.node 16097 (.node 15803 (.node 15731 .empty .empty) .empty) (.node 17021 .empty .empty)) (.node 17189 (.node 17183 .empty .empty) (.node 20177 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5351_ordered : SortedNatTree.ordered row5351 = true := by decide
set_option maxRecDepth 4096 in
theorem row5351_values : SortedNatTree.values row5351 = Data.row5351 := rfl
theorem row5351_checked : SortedNatTree.contains row5351 = Data.row5351.contains := by
  rw [SortedNatTree.contains_eq_list row5351 row5351_ordered, row5351_values]
def row5381 : SortedNatTree.Tree := (.node 12503 (.node 8081 (.node 6599 (.node 5981 (.node 5507 .empty .empty) (.node 6011 .empty .empty)) (.node 7877 (.node 7793 .empty .empty) (.node 7919 .empty .empty))) (.node 10289 (.node 9851 (.node 9791 .empty .empty) (.node 9929 .empty .empty)) (.node 11549 (.node 10883 .empty .empty) (.node 12269 .empty .empty)))) (.node 16001 (.node 14411 (.node 13613 (.node 13229 .empty .empty) (.node 13931 .empty .empty)) (.node 15581 (.node 14747 .empty .empty) (.node 15749 .empty .empty))) (.node 19997 (.node 17837 (.node 16823 .empty .empty) (.node 19073 .empty .empty)) (.node 20327 (.node 20123 .empty .empty) (.node 20333 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5381_ordered : SortedNatTree.ordered row5381 = true := by decide
set_option maxRecDepth 4096 in
theorem row5381_values : SortedNatTree.values row5381 = Data.row5381 := rfl
theorem row5381_checked : SortedNatTree.contains row5381 = Data.row5381.contains := by
  rw [SortedNatTree.contains_eq_list row5381 row5381_ordered, row5381_values]
def row5387 : SortedNatTree.Tree := (.node 10181 (.node 7487 (.node 6353 (.node 5807 (.node 5399 .empty .empty) (.node 6269 .empty .empty)) (.node 6959 (.node 6653 .empty .empty) (.node 7127 .empty .empty))) (.node 9689 (.node 9467 (.node 9173 .empty .empty) (.node 9479 .empty .empty)) (.node 10091 (.node 9857 .empty .empty) .empty))) (.node 16223 (.node 14327 (.node 12569 (.node 10463 .empty .empty) (.node 13127 .empty .empty)) (.node 15083 (.node 14753 .empty .empty) .empty)) (.node 17903 (.node 17189 (.node 16901 .empty .empty) (.node 17789 .empty .empty)) (.node 18839 (.node 18041 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5387_ordered : SortedNatTree.ordered row5387 = true := by decide
set_option maxRecDepth 4096 in
theorem row5387_values : SortedNatTree.values row5387 = Data.row5387 := rfl
theorem row5387_checked : SortedNatTree.contains row5387 = Data.row5387.contains := by
  rw [SortedNatTree.contains_eq_list row5387 row5387_ordered, row5387_values]
def row5393 : SortedNatTree.Tree := (.node 14549 (.node 10133 (.node 7817 (.node 6521 (.node 5669 .empty .empty) .empty) (.node 8369 (.node 7937 .empty .empty) .empty)) (.node 12491 (.node 12161 (.node 11027 .empty .empty) .empty) (.node 13751 .empty .empty))) (.node 17159 (.node 16187 (.node 15473 (.node 15131 .empty .empty) .empty) (.node 16979 (.node 16301 .empty .empty) .empty)) (.node 19463 (.node 19457 (.node 18617 .empty .empty) .empty) (.node 19553 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5393_ordered : SortedNatTree.ordered row5393 = true := by decide
set_option maxRecDepth 4096 in
theorem row5393_values : SortedNatTree.values row5393 = Data.row5393 := rfl
theorem row5393_checked : SortedNatTree.contains row5393 = Data.row5393.contains := by
  rw [SortedNatTree.contains_eq_list row5393 row5393_ordered, row5393_values]
def row5399 : SortedNatTree.Tree := (.node 11699 (.node 8117 (.node 7247 (.node 6473 (.node 6329 .empty .empty) (.node 7079 .empty .empty)) (.node 7649 (.node 7523 .empty .empty) (.node 8111 .empty .empty))) (.node 10037 (.node 9431 (.node 9371 .empty .empty) (.node 9749 .empty .empty)) (.node 11069 (.node 10883 .empty .empty) (.node 11423 .empty .empty)))) (.node 15749 (.node 15269 (.node 13469 (.node 13259 .empty .empty) (.node 13883 .empty .empty)) (.node 15401 (.node 15287 .empty .empty) (.node 15443 .empty .empty))) (.node 18131 (.node 15791 (.node 15773 .empty .empty) (.node 16193 .empty .empty)) (.node 20411 (.node 18149 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5399_ordered : SortedNatTree.ordered row5399 = true := by decide
set_option maxRecDepth 4096 in
theorem row5399_values : SortedNatTree.values row5399 = Data.row5399 := rfl
theorem row5399_checked : SortedNatTree.contains row5399 = Data.row5399.contains := by
  rw [SortedNatTree.contains_eq_list row5399 row5399_ordered, row5399_values]
def row5407 : SortedNatTree.Tree := (.node 13963 (.node 9391 (.node 6163 (.node 5743 (.node 5449 .empty .empty) (.node 5779 .empty .empty)) (.node 8287 (.node 6961 .empty .empty) (.node 8461 .empty .empty))) (.node 10957 (.node 10567 (.node 9679 .empty .empty) (.node 10909 .empty .empty)) (.node 12301 (.node 11113 .empty .empty) .empty))) (.node 18181 (.node 15937 (.node 15199 (.node 14341 .empty .empty) (.node 15313 .empty .empty)) (.node 17659 (.node 16921 .empty .empty) .empty)) (.node 19507 (.node 18637 (.node 18301 .empty .empty) (.node 19429 .empty .empty)) (.node 20347 (.node 20143 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5407_ordered : SortedNatTree.ordered row5407 = true := by decide
set_option maxRecDepth 4096 in
theorem row5407_values : SortedNatTree.values row5407 = Data.row5407 := rfl
theorem row5407_checked : SortedNatTree.contains row5407 = Data.row5407.contains := by
  rw [SortedNatTree.contains_eq_list row5407 row5407_ordered, row5407_values]
def row5413 : SortedNatTree.Tree := (.node 13411 (.node 8209 (.node 5821 (.node 5569 (.node 5527 .empty .empty) .empty) (.node 7159 (.node 6451 .empty .empty) .empty)) (.node 10477 (.node 9697 (.node 9619 .empty .empty) .empty) (.node 12241 (.node 11149 .empty .empty) .empty))) (.node 16333 (.node 14251 (.node 13999 (.node 13789 .empty .empty) .empty) (.node 15559 (.node 15259 .empty .empty) .empty)) (.node 19081 (.node 18313 (.node 17977 .empty .empty) .empty) (.node 19543 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5413_ordered : SortedNatTree.ordered row5413 = true := by decide
set_option maxRecDepth 4096 in
theorem row5413_values : SortedNatTree.values row5413 = Data.row5413 := rfl
theorem row5413_checked : SortedNatTree.contains row5413 = Data.row5413.contains := by
  rw [SortedNatTree.contains_eq_list row5413 row5413_ordered, row5413_values]
def row5417 : SortedNatTree.Tree := (.node 13229 (.node 9239 (.node 8513 (.node 7001 (.node 6131 (.node 5981 .empty .empty) .empty) (.node 8219 .empty .empty)) (.node 8963 (.node 8819 (.node 8627 .empty .empty) .empty) (.node 9029 .empty .empty))) (.node 11813 (.node 9689 (.node 9533 (.node 9371 .empty .empty) .empty) (.node 10331 .empty .empty)) (.node 13037 (.node 12653 .empty .empty) (.node 13187 .empty .empty)))) (.node 17021 (.node 14951 (.node 14549 (.node 14489 (.node 14081 .empty .empty) .empty) (.node 14627 .empty .empty)) (.node 15413 (.node 15401 .empty .empty) (.node 15959 .empty .empty))) (.node 19121 (.node 18059 (.node 17783 (.node 17093 .empty .empty) .empty) (.node 18371 .empty .empty)) (.node 20129 (.node 20117 .empty .empty) (.node 20549 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5417_ordered : SortedNatTree.ordered row5417 = true := by decide
set_option maxRecDepth 4096 in
theorem row5417_values : SortedNatTree.values row5417 = Data.row5417 := rfl
theorem row5417_checked : SortedNatTree.contains row5417 = Data.row5417.contains := by
  rw [SortedNatTree.contains_eq_list row5417 row5417_ordered, row5417_values]
def row5419 : SortedNatTree.Tree := (.node 13381 (.node 10627 (.node 8803 (.node 8527 (.node 7687 .empty .empty) (.node 8719 .empty .empty)) (.node 9421 (.node 9241 .empty .empty) (.node 10009 .empty .empty))) (.node 11467 (.node 11239 (.node 10657 .empty .empty) (.node 11353 .empty .empty)) (.node 12739 (.node 12601 .empty .empty) (.node 13093 .empty .empty)))) (.node 17569 (.node 14683 (.node 14011 (.node 13399 .empty .empty) (.node 14533 .empty .empty)) (.node 15331 (.node 15091 .empty .empty) (.node 15889 .empty .empty))) (.node 19051 (.node 18301 (.node 17713 .empty .empty) (.node 18427 .empty .empty)) (.node 20521 (.node 20107 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5419_ordered : SortedNatTree.ordered row5419 = true := by decide
set_option maxRecDepth 4096 in
theorem row5419_values : SortedNatTree.values row5419 = Data.row5419 := rfl
theorem row5419_checked : SortedNatTree.contains row5419 = Data.row5419.contains := by
  rw [SortedNatTree.contains_eq_list row5419 row5419_ordered, row5419_values]
def row5431 : SortedNatTree.Tree := (.node 11383 (.node 8167 (.node 7243 (.node 5743 (.node 5641 .empty .empty) (.node 6151 .empty .empty)) (.node 7573 (.node 7507 .empty .empty) .empty)) (.node 10357 (.node 9649 (.node 8599 .empty .empty) (.node 9973 .empty .empty)) (.node 11131 (.node 10639 .empty .empty) .empty))) (.node 16477 (.node 15259 (.node 13723 (.node 12541 .empty .empty) (.node 14251 .empty .empty)) (.node 16033 (.node 15919 .empty .empty) .empty)) (.node 18493 (.node 17569 (.node 17431 .empty .empty) .empty) (.node 18913 (.node 18679 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5431_ordered : SortedNatTree.ordered row5431 = true := by decide
set_option maxRecDepth 4096 in
theorem row5431_values : SortedNatTree.values row5431 = Data.row5431 := rfl
theorem row5431_checked : SortedNatTree.contains row5431 = Data.row5431.contains := by
  rw [SortedNatTree.contains_eq_list row5431 row5431_ordered, row5431_values]
def row5437 : SortedNatTree.Tree := (.node 14071 (.node 9733 (.node 8017 (.node 6961 (.node 5689 .empty .empty) (.node 7159 .empty .empty)) (.node 9001 (.node 8263 .empty .empty) (.node 9067 .empty .empty))) (.node 11317 (.node 10753 (.node 9883 .empty .empty) (.node 10831 .empty .empty)) (.node 11863 (.node 11719 .empty .empty) (.node 13921 .empty .empty)))) (.node 18289 (.node 15919 (.node 15121 (.node 14629 .empty .empty) (.node 15643 .empty .empty)) (.node 16267 (.node 16183 .empty .empty) (.node 18223 .empty .empty))) (.node 20011 (.node 18661 (.node 18541 .empty .empty) (.node 18913 .empty .empty)) (.node 20533 (.node 20407 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5437_ordered : SortedNatTree.ordered row5437 = true := by decide
set_option maxRecDepth 4096 in
theorem row5437_values : SortedNatTree.values row5437 = Data.row5437 := rfl
theorem row5437_checked : SortedNatTree.contains row5437 = Data.row5437.contains := by
  rw [SortedNatTree.contains_eq_list row5437 row5437_ordered, row5437_values]
def row5441 : SortedNatTree.Tree := (.node 11657 (.node 7457 (.node 5849 (.node 5717 (.node 5693 .empty .empty) (.node 5783 .empty .empty)) (.node 6959 (.node 6287 .empty .empty) (.node 7019 .empty .empty))) (.node 10067 (.node 9623 (.node 7547 .empty .empty) (.node 9767 .empty .empty)) (.node 11423 (.node 10859 .empty .empty) .empty))) (.node 13799 (.node 13103 (.node 12539 (.node 11927 .empty .empty) (.node 13049 .empty .empty)) (.node 13463 (.node 13151 .empty .empty) .empty)) (.node 17393 (.node 15149 (.node 13841 .empty .empty) (.node 16787 .empty .empty)) (.node 19937 (.node 19013 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5441_ordered : SortedNatTree.ordered row5441 = true := by decide
set_option maxRecDepth 4096 in
theorem row5441_values : SortedNatTree.values row5441 = Data.row5441 := rfl
theorem row5441_checked : SortedNatTree.contains row5441 = Data.row5441.contains := by
  rw [SortedNatTree.contains_eq_list row5441 row5441_ordered, row5441_values]
def row5443 : SortedNatTree.Tree := (.node 10687 (.node 7039 (.node 6379 (.node 5923 (.node 5791 (.node 5653 .empty .empty) .empty) (.node 6079 .empty .empty)) (.node 6637 (.node 6553 .empty .empty) (.node 6841 .empty .empty))) (.node 8101 (.node 7549 (.node 7393 (.node 7351 .empty .empty) .empty) (.node 7561 .empty .empty)) (.node 9067 (.node 8893 .empty .empty) (.node 9391 .empty .empty)))) (.node 15559 (.node 12829 (.node 12547 (.node 12343 (.node 12037 .empty .empty) .empty) (.node 12799 .empty .empty)) (.node 13723 (.node 13513 .empty .empty) (.node 14431 .empty .empty))) (.node 18793 (.node 16651 (.node 15859 .empty .empty) (.node 17077 .empty .empty)) (.node 20353 (.node 19801 .empty .empty) (.node 20479 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5443_ordered : SortedNatTree.ordered row5443 = true := by decide
set_option maxRecDepth 4096 in
theorem row5443_values : SortedNatTree.values row5443 = Data.row5443 := rfl
theorem row5443_checked : SortedNatTree.contains row5443 = Data.row5443.contains := by
  rw [SortedNatTree.contains_eq_list row5443 row5443_ordered, row5443_values]
def row5449 : SortedNatTree.Tree := (.node 13093 (.node 8581 (.node 7507 (.node 7243 (.node 6073 (.node 5851 .empty .empty) .empty) (.node 7321 .empty .empty)) (.node 8089 (.node 7549 .empty .empty) (.node 8539 .empty .empty))) (.node 11329 (.node 9817 (.node 8929 (.node 8623 .empty .empty) .empty) (.node 10159 .empty .empty)) (.node 11779 (.node 11701 .empty .empty) (.node 12073 .empty .empty)))) (.node 15619 (.node 14011 (.node 13591 (.node 13267 (.node 13249 .empty .empty) .empty) (.node 13627 .empty .empty)) (.node 14983 (.node 14389 .empty .empty) (.node 15139 .empty .empty))) (.node 18097 (.node 17029 (.node 15901 (.node 15643 .empty .empty) .empty) (.node 17239 .empty .empty)) (.node 19429 (.node 18457 .empty .empty) (.node 20029 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5449_ordered : SortedNatTree.ordered row5449 = true := by decide
set_option maxRecDepth 4096 in
theorem row5449_values : SortedNatTree.values row5449 = Data.row5449 := rfl
theorem row5449_checked : SortedNatTree.contains row5449 = Data.row5449.contains := by
  rw [SortedNatTree.contains_eq_list row5449 row5449_ordered, row5449_values]
def row5471 : SortedNatTree.Tree := (.node 12113 (.node 6653 (.node 6203 (.node 5807 (.node 5783 .empty .empty) (.node 6197 .empty .empty)) (.node 6491 (.node 6449 .empty .empty) .empty)) (.node 8171 (.node 7841 (.node 7583 .empty .empty) .empty) (.node 11597 (.node 11519 .empty .empty) .empty))) (.node 17033 (.node 15731 (.node 13001 (.node 12647 .empty .empty) (.node 13751 .empty .empty)) (.node 16061 (.node 15797 .empty .empty) .empty)) (.node 18329 (.node 18233 (.node 17903 .empty .empty) .empty) (.node 20549 (.node 19373 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5471_ordered : SortedNatTree.ordered row5471 = true := by decide
set_option maxRecDepth 4096 in
theorem row5471_values : SortedNatTree.values row5471 = Data.row5471 := rfl
theorem row5471_checked : SortedNatTree.contains row5471 = Data.row5471.contains := by
  rw [SortedNatTree.contains_eq_list row5471 row5471_ordered, row5471_values]
def row5477 : SortedNatTree.Tree := (.node 11447 (.node 8081 (.node 6311 (.node 6053 (.node 5927 (.node 5711 .empty .empty) .empty) (.node 6089 .empty .empty)) (.node 7109 (.node 6599 .empty .empty) (.node 8039 .empty .empty))) (.node 9041 (.node 8741 (.node 8123 .empty .empty) (.node 8951 .empty .empty)) (.node 9677 (.node 9551 .empty .empty) (.node 9839 .empty .empty)))) (.node 15263 (.node 13499 (.node 12227 (.node 11981 .empty .empty) (.node 13457 .empty .empty)) (.node 14033 (.node 14009 .empty .empty) (.node 14423 .empty .empty))) (.node 17939 (.node 16229 (.node 15803 .empty .empty) (.node 16979 .empty .empty)) (.node 19577 (.node 18839 .empty .empty) (.node 19709 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5477_ordered : SortedNatTree.ordered row5477 = true := by decide
set_option maxRecDepth 4096 in
theorem row5477_values : SortedNatTree.values row5477 = Data.row5477 := rfl
theorem row5477_checked : SortedNatTree.contains row5477 = Data.row5477.contains := by
  rw [SortedNatTree.contains_eq_list row5477 row5477_ordered, row5477_values]
def row5479 : SortedNatTree.Tree := (.node 10501 (.node 7927 (.node 5869 (.node 5827 (.node 5581 .empty .empty) (.node 5857 .empty .empty)) (.node 6607 (.node 6271 .empty .empty) (.node 7321 .empty .empty))) (.node 9817 (.node 9127 (.node 8209 .empty .empty) (.node 9649 .empty .empty)) (.node 10453 (.node 9973 .empty .empty) .empty))) (.node 17107 (.node 13669 (.node 11779 (.node 11311 .empty .empty) (.node 13537 .empty .empty)) (.node 14797 (.node 13831 .empty .empty) (.node 16063 .empty .empty))) (.node 19081 (.node 17449 (.node 17239 .empty .empty) (.node 17659 .empty .empty)) (.node 19963 (.node 19759 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5479_ordered : SortedNatTree.ordered row5479 = true := by decide
set_option maxRecDepth 4096 in
theorem row5479_values : SortedNatTree.values row5479 = Data.row5479 := rfl
theorem row5479_checked : SortedNatTree.contains row5479 = Data.row5479.contains := by
  rw [SortedNatTree.contains_eq_list row5479 row5479_ordered, row5479_values]
def row5483 : SortedNatTree.Tree := (.node 11321 (.node 7487 (.node 6491 (.node 5987 (.node 5507 .empty .empty) (.node 6029 .empty .empty)) (.node 7247 (.node 7103 .empty .empty) .empty)) (.node 8423 (.node 8369 (.node 8297 .empty .empty) .empty) (.node 10463 (.node 9479 .empty .empty) .empty))) (.node 15227 (.node 13757 (.node 13463 (.node 12437 .empty .empty) .empty) (.node 14813 (.node 14387 .empty .empty) .empty)) (.node 16673 (.node 16361 (.node 15797 .empty .empty) .empty) (.node 18899 (.node 17729 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5483_ordered : SortedNatTree.ordered row5483 = true := by decide
set_option maxRecDepth 4096 in
theorem row5483_values : SortedNatTree.values row5483 = Data.row5483 := rfl
theorem row5483_checked : SortedNatTree.contains row5483 = Data.row5483.contains := by
  rw [SortedNatTree.contains_eq_list row5483 row5483_ordered, row5483_values]
def row5501 : SortedNatTree.Tree := (.node 11777 (.node 8573 (.node 6359 (.node 5867 (.node 5591 .empty .empty) (.node 5879 .empty .empty)) (.node 7349 (.node 7211 .empty .empty) (.node 7481 .empty .empty))) (.node 11117 (.node 10289 (.node 9491 .empty .empty) (.node 10853 .empty .empty)) (.node 11519 (.node 11273 .empty .empty) .empty))) (.node 17417 (.node 14249 (.node 12653 (.node 12113 .empty .empty) (.node 13163 .empty .empty)) (.node 14843 (.node 14627 .empty .empty) (.node 16223 .empty .empty))) (.node 19571 (.node 18311 (.node 17783 .empty .empty) (.node 19433 .empty .empty)) (.node 20399 (.node 19991 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5501_ordered : SortedNatTree.ordered row5501 = true := by decide
set_option maxRecDepth 4096 in
theorem row5501_values : SortedNatTree.values row5501 = Data.row5501 := rfl
theorem row5501_checked : SortedNatTree.contains row5501 = Data.row5501.contains := by
  rw [SortedNatTree.contains_eq_list row5501 row5501_ordered, row5501_values]
def row5503 : SortedNatTree.Tree := (.node 12289 (.node 8353 (.node 7393 (.node 6469 (.node 5821 .empty .empty) (.node 7027 .empty .empty)) (.node 8311 (.node 7603 .empty .empty) .empty)) (.node 9967 (.node 8887 (.node 8521 .empty .empty) .empty) (.node 11959 (.node 10093 .empty .empty) .empty))) (.node 15679 (.node 14437 (.node 13693 (.node 12541 .empty .empty) .empty) (.node 15607 (.node 15061 .empty .empty) .empty)) (.node 16411 (.node 16369 (.node 16141 .empty .empty) .empty) (.node 17107 (.node 16573 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5503_ordered : SortedNatTree.ordered row5503 = true := by decide
set_option maxRecDepth 4096 in
theorem row5503_values : SortedNatTree.values row5503 = Data.row5503 := rfl
theorem row5503_checked : SortedNatTree.contains row5503 = Data.row5503.contains := by
  rw [SortedNatTree.contains_eq_list row5503 row5503_ordered, row5503_values]
def row5507 : SortedNatTree.Tree := (.node 11717 (.node 7853 (.node 6257 (.node 6173 (.node 5939 .empty .empty) (.node 6221 .empty .empty)) (.node 7643 (.node 6803 .empty .empty) (.node 7829 .empty .empty))) (.node 10163 (.node 8291 (.node 7877 .empty .empty) (.node 9623 .empty .empty)) (.node 10607 (.node 10169 .empty .empty) (.node 10739 .empty .empty)))) (.node 15413 (.node 13859 (.node 12689 (.node 12647 .empty .empty) (.node 13469 .empty .empty)) (.node 15161 (.node 14831 .empty .empty) (.node 15377 .empty .empty))) (.node 17891 (.node 16091 (.node 15773 .empty .empty) (.node 16487 .empty .empty)) (.node 19139 (.node 18731 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5507_ordered : SortedNatTree.ordered row5507 = true := by decide
set_option maxRecDepth 4096 in
theorem row5507_values : SortedNatTree.values row5507 = Data.row5507 := rfl
theorem row5507_checked : SortedNatTree.contains row5507 = Data.row5507.contains := by
  rw [SortedNatTree.contains_eq_list row5507 row5507_ordered, row5507_values]
def row5519 : SortedNatTree.Tree := (.node 13697 (.node 10847 (.node 8081 (.node 6299 (.node 5711 .empty .empty) (.node 7517 .empty .empty)) (.node 8999 (.node 8753 .empty .empty) (.node 9533 .empty .empty))) (.node 12269 (.node 11483 (.node 11027 .empty .empty) (.node 12143 .empty .empty)) (.node 13331 (.node 12821 .empty .empty) .empty))) (.node 17117 (.node 15809 (.node 15263 (.node 13781 .empty .empty) (.node 15773 .empty .empty)) (.node 16433 (.node 16229 .empty .empty) (.node 16691 .empty .empty))) (.node 19763 (.node 18503 (.node 17609 .empty .empty) (.node 19079 .empty .empty)) (.node 20297 (.node 20261 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5519_ordered : SortedNatTree.ordered row5519 = true := by decide
set_option maxRecDepth 4096 in
theorem row5519_values : SortedNatTree.values row5519 = Data.row5519 := rfl
theorem row5519_checked : SortedNatTree.contains row5519 = Data.row5519.contains := by
  rw [SortedNatTree.contains_eq_list row5519 row5519_ordered, row5519_values]
def row5521 : SortedNatTree.Tree := (.node 14821 (.node 10687 (.node 7951 (.node 5851 (.node 5701 .empty .empty) (.node 7159 .empty .empty)) (.node 9547 (.node 9403 .empty .empty) .empty)) (.node 12487 (.node 11311 (.node 11257 .empty .empty) (.node 11383 .empty .empty)) (.node 14071 (.node 12721 .empty .empty) .empty))) (.node 17191 (.node 16333 (.node 15583 (.node 15373 .empty .empty) (.node 16141 .empty .empty)) (.node 16921 (.node 16651 .empty .empty) .empty)) (.node 18493 (.node 17659 (.node 17299 .empty .empty) .empty) (.node 19819 (.node 19753 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5521_ordered : SortedNatTree.ordered row5521 = true := by decide
set_option maxRecDepth 4096 in
theorem row5521_values : SortedNatTree.values row5521 = Data.row5521 := rfl
theorem row5521_checked : SortedNatTree.contains row5521 = Data.row5521.contains := by
  rw [SortedNatTree.contains_eq_list row5521 row5521_ordered, row5521_values]
def row5527 : SortedNatTree.Tree := (.node 11839 (.node 8719 (.node 7129 (.node 6577 (.node 6343 .empty .empty) (.node 6619 .empty .empty)) (.node 7753 (.node 7549 .empty .empty) .empty)) (.node 9871 (.node 9403 (.node 9349 .empty .empty) .empty) (.node 11329 (.node 10867 .empty .empty) .empty))) (.node 14983 (.node 13417 (.node 12409 (.node 12157 .empty .empty) .empty) (.node 13687 (.node 13669 .empty .empty) .empty)) (.node 16741 (.node 15349 (.node 15313 .empty .empty) .empty) (.node 20101 (.node 16987 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5527_ordered : SortedNatTree.ordered row5527 = true := by decide
set_option maxRecDepth 4096 in
theorem row5527_values : SortedNatTree.values row5527 = Data.row5527 := rfl
theorem row5527_checked : SortedNatTree.contains row5527 = Data.row5527.contains := by
  rw [SortedNatTree.contains_eq_list row5527 row5527_ordered, row5527_values]
def row5531 : SortedNatTree.Tree := (.node 12473 (.node 8387 (.node 7673 (.node 7121 (.node 6089 .empty .empty) .empty) (.node 8009 (.node 7883 .empty .empty) .empty)) (.node 10391 (.node 10193 (.node 9851 .empty .empty) .empty) (.node 11549 (.node 10667 .empty .empty) .empty))) (.node 15971 (.node 13007 (.node 12821 (.node 12713 .empty .empty) .empty) (.node 15131 (.node 14087 .empty .empty) .empty)) (.node 16451 (.node 16253 (.node 16127 .empty .empty) .empty) (.node 20411 (.node 18803 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5531_ordered : SortedNatTree.ordered row5531 = true := by decide
set_option maxRecDepth 4096 in
theorem row5531_values : SortedNatTree.values row5531 = Data.row5531 := rfl
theorem row5531_checked : SortedNatTree.contains row5531 = Data.row5531.contains := by
  rw [SortedNatTree.contains_eq_list row5531 row5531_ordered, row5531_values]
def row5557 : SortedNatTree.Tree := (.node 10627 (.node 8053 (.node 7507 (.node 6529 (.node 6067 .empty .empty) (.node 7393 .empty .empty)) (.node 7789 (.node 7537 .empty .empty) .empty)) (.node 8707 (.node 8233 (.node 8167 .empty .empty) (.node 8293 .empty .empty)) (.node 10069 (.node 9133 .empty .empty) .empty))) (.node 14407 (.node 13339 (.node 11437 (.node 11197 .empty .empty) (.node 11941 .empty .empty)) (.node 14197 (.node 13723 .empty .empty) .empty)) (.node 16561 (.node 15289 (.node 14461 .empty .empty) (.node 15373 .empty .empty)) (.node 17959 (.node 16831 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5557_ordered : SortedNatTree.ordered row5557 = true := by decide
set_option maxRecDepth 4096 in
theorem row5557_values : SortedNatTree.values row5557 = Data.row5557 := rfl
theorem row5557_checked : SortedNatTree.contains row5557 = Data.row5557.contains := by
  rw [SortedNatTree.contains_eq_list row5557 row5557_ordered, row5557_values]
def row5563 : SortedNatTree.Tree := (.node 12241 (.node 8623 (.node 6781 (.node 6037 (.node 5791 .empty .empty) (.node 6079 .empty .empty)) (.node 8053 (.node 7951 .empty .empty) .empty)) (.node 10459 (.node 10303 (.node 9817 .empty .empty) .empty) (.node 12157 (.node 11779 .empty .empty) .empty))) (.node 14947 (.node 13567 (.node 12703 (.node 12637 .empty .empty) (.node 12781 .empty .empty)) (.node 14923 (.node 14173 .empty .empty) .empty)) (.node 15289 (.node 15217 (.node 15031 .empty .empty) .empty) (.node 16417 (.node 15493 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5563_ordered : SortedNatTree.ordered row5563 = true := by decide
set_option maxRecDepth 4096 in
theorem row5563_values : SortedNatTree.values row5563 = Data.row5563 := rfl
theorem row5563_checked : SortedNatTree.contains row5563 = Data.row5563.contains := by
  rw [SortedNatTree.contains_eq_list row5563 row5563_ordered, row5563_values]
def row5569 : SortedNatTree.Tree := (.node 11119 (.node 7309 (.node 6451 (.node 6373 (.node 5923 .empty .empty) .empty) (.node 6577 (.node 6469 .empty .empty) .empty)) (.node 8887 (.node 8521 (.node 7687 .empty .empty) .empty) (.node 10267 (.node 9433 .empty .empty) .empty))) (.node 16033 (.node 12511 (.node 12289 (.node 12007 .empty .empty) .empty) (.node 13309 (.node 13177 .empty .empty) .empty)) (.node 19717 (.node 18217 (.node 18133 .empty .empty) .empty) (.node 20113 (.node 20107 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5569_ordered : SortedNatTree.ordered row5569 = true := by decide
set_option maxRecDepth 4096 in
theorem row5569_values : SortedNatTree.values row5569 = Data.row5569 := rfl
theorem row5569_checked : SortedNatTree.contains row5569 = Data.row5569.contains := by
  rw [SortedNatTree.contains_eq_list row5569 row5569_ordered, row5569_values]
def row5573 : SortedNatTree.Tree := (.node 12281 (.node 9587 (.node 8219 (.node 7457 (.node 6833 .empty .empty) (.node 7937 .empty .empty)) (.node 8849 (.node 8537 .empty .empty) .empty)) (.node 11423 (.node 10433 (.node 10037 .empty .empty) (.node 10781 .empty .empty)) (.node 11903 (.node 11549 .empty .empty) .empty))) (.node 16103 (.node 14543 (.node 13103 (.node 12527 .empty .empty) (.node 13997 .empty .empty)) (.node 15683 (.node 14969 .empty .empty) .empty)) (.node 19163 (.node 17627 (.node 17489 .empty .empty) .empty) (.node 20441 (.node 20411 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5573_ordered : SortedNatTree.ordered row5573 = true := by decide
set_option maxRecDepth 4096 in
theorem row5573_values : SortedNatTree.values row5573 = Data.row5573 := rfl
theorem row5573_checked : SortedNatTree.contains row5573 = Data.row5573.contains := by
  rw [SortedNatTree.contains_eq_list row5573 row5573_ordered, row5573_values]
def row5581 : SortedNatTree.Tree := (.node 14107 (.node 8017 (.node 6067 (.node 5737 (.node 5653 .empty .empty) (.node 5839 .empty .empty)) (.node 7351 (.node 6991 .empty .empty) (.node 7561 .empty .empty))) (.node 11551 (.node 9349 (.node 8101 .empty .empty) (.node 9949 .empty .empty)) (.node 13327 (.node 12697 .empty .empty) .empty))) (.node 17257 (.node 16567 (.node 16381 (.node 15391 .empty .empty) (.node 16519 .empty .empty)) (.node 17107 (.node 16759 .empty .empty) .empty)) (.node 19249 (.node 18061 (.node 17467 .empty .empty) (.node 18793 .empty .empty)) (.node 20161 (.node 19423 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5581_ordered : SortedNatTree.ordered row5581 = true := by decide
set_option maxRecDepth 4096 in
theorem row5581_values : SortedNatTree.values row5581 = Data.row5581 := rfl
theorem row5581_checked : SortedNatTree.contains row5581 = Data.row5581.contains := by
  rw [SortedNatTree.contains_eq_list row5581 row5581_ordered, row5581_values]
def row5591 : SortedNatTree.Tree := (.node 13757 (.node 10529 (.node 8609 (.node 6701 (.node 6173 .empty .empty) (.node 6737 .empty .empty)) (.node 9491 (.node 8969 .empty .empty) (.node 9887 .empty .empty))) (.node 12041 (.node 11279 (.node 10949 .empty .empty) (.node 11867 .empty .empty)) (.node 13259 (.node 12899 .empty .empty) (.node 13553 .empty .empty)))) (.node 16931 (.node 15269 (.node 14951 (.node 14639 .empty .empty) (.node 15233 .empty .empty)) (.node 16529 (.node 15731 .empty .empty) (.node 16829 .empty .empty))) (.node 19013 (.node 17939 (.node 17387 .empty .empty) (.node 18401 .empty .empty)) (.node 20333 (.node 19991 .empty .empty) (.node 20399 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5591_ordered : SortedNatTree.ordered row5591 = true := by decide
set_option maxRecDepth 4096 in
theorem row5591_values : SortedNatTree.values row5591 = Data.row5591 := rfl
theorem row5591_checked : SortedNatTree.contains row5591 = Data.row5591.contains := by
  rw [SortedNatTree.contains_eq_list row5591 row5591_ordered, row5591_values]
def row5623 : SortedNatTree.Tree := (.node 11353 (.node 8311 (.node 6907 (.node 6421 (.node 5821 .empty .empty) (.node 6547 .empty .empty)) (.node 7639 (.node 6949 .empty .empty) (.node 7681 .empty .empty))) (.node 9769 (.node 9319 (.node 8317 .empty .empty) (.node 9433 .empty .empty)) (.node 10093 (.node 10009 .empty .empty) (.node 10369 .empty .empty)))) (.node 16567 (.node 13831 (.node 13441 (.node 11887 .empty .empty) (.node 13789 .empty .empty)) (.node 15217 (.node 14449 .empty .empty) (.node 16087 .empty .empty))) (.node 18457 (.node 17761 (.node 16693 .empty .empty) (.node 18121 .empty .empty)) (.node 19717 (.node 19531 .empty .empty) (.node 19861 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5623_ordered : SortedNatTree.ordered row5623 = true := by decide
set_option maxRecDepth 4096 in
theorem row5623_values : SortedNatTree.values row5623 = Data.row5623 := rfl
theorem row5623_checked : SortedNatTree.contains row5623 = Data.row5623.contains := by
  rw [SortedNatTree.contains_eq_list row5623 row5623_ordered, row5623_values]
def row5639 : SortedNatTree.Tree := (.node 12539 (.node 8039 (.node 6833 (.node 6737 (.node 6563 .empty .empty) (.node 6827 .empty .empty)) (.node 7079 (.node 6983 .empty .empty) (.node 7877 .empty .empty))) (.node 9839 (.node 9311 (.node 8963 .empty .empty) (.node 9377 .empty .empty)) (.node 10631 (.node 10103 .empty .empty) .empty))) (.node 14423 (.node 13487 (.node 13151 (.node 12653 .empty .empty) (.node 13451 .empty .empty)) (.node 13841 (.node 13619 .empty .empty) .empty)) (.node 18287 (.node 15227 (.node 14969 .empty .empty) (.node 16007 .empty .empty)) (.node 20297 (.node 19469 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5639_ordered : SortedNatTree.ordered row5639 = true := by decide
set_option maxRecDepth 4096 in
theorem row5639_values : SortedNatTree.values row5639 = Data.row5639 := rfl
theorem row5639_checked : SortedNatTree.contains row5639 = Data.row5639.contains := by
  rw [SortedNatTree.contains_eq_list row5639 row5639_ordered, row5639_values]
def row5641 : SortedNatTree.Tree := (.node 11173 (.node 7759 (.node 6793 (.node 6007 (.node 5743 .empty .empty) (.node 6637 .empty .empty)) (.node 7687 (.node 7027 .empty .empty) .empty)) (.node 9091 (.node 8209 (.node 8191 .empty .empty) .empty) (.node 10753 (.node 9463 .empty .empty) .empty))) (.node 15991 (.node 13513 (.node 12799 (.node 12163 .empty .empty) (.node 13093 .empty .empty)) (.node 15907 (.node 15901 .empty .empty) .empty)) (.node 18133 (.node 17707 (.node 17293 .empty .empty) .empty) (.node 20023 (.node 19447 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5641_ordered : SortedNatTree.ordered row5641 = true := by decide
set_option maxRecDepth 4096 in
theorem row5641_values : SortedNatTree.values row5641 = Data.row5641 := rfl
theorem row5641_checked : SortedNatTree.contains row5641 = Data.row5641.contains := by
  rw [SortedNatTree.contains_eq_list row5641 row5641_ordered, row5641_values]
def row5647 : SortedNatTree.Tree := (.node 11311 (.node 9043 (.node 7129 (.node 6373 (.node 5659 .empty .empty) (.node 6733 .empty .empty)) (.node 7867 (.node 7741 .empty .empty) .empty)) (.node 9931 (.node 9817 (.node 9403 .empty .empty) .empty) (.node 11299 (.node 10141 .empty .empty) .empty))) (.node 17467 (.node 14779 (.node 13729 (.node 12391 .empty .empty) .empty) (.node 17047 (.node 15331 .empty .empty) .empty)) (.node 18523 (.node 18397 (.node 18313 .empty .empty) .empty) (.node 20113 (.node 19489 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5647_ordered : SortedNatTree.ordered row5647 = true := by decide
set_option maxRecDepth 4096 in
theorem row5647_values : SortedNatTree.values row5647 = Data.row5647 := rfl
theorem row5647_checked : SortedNatTree.contains row5647 = Data.row5647.contains := by
  rw [SortedNatTree.contains_eq_list row5647 row5647_ordered, row5647_values]
def row5651 : SortedNatTree.Tree := (.node 13553 (.node 9311 (.node 7691 (.node 6011 (.node 5783 .empty .empty) (.node 7499 .empty .empty)) (.node 8663 (.node 8093 .empty .empty) (.node 8831 .empty .empty))) (.node 11321 (.node 9629 (.node 9413 .empty .empty) (.node 10589 .empty .empty)) (.node 13259 (.node 12959 .empty .empty) .empty))) (.node 18839 (.node 14747 (.node 14657 (.node 14327 .empty .empty) (.node 14741 .empty .empty)) (.node 16787 (.node 15149 .empty .empty) .empty)) (.node 19853 (.node 19427 (.node 19013 .empty .empty) (.node 19571 .empty .empty)) (.node 20123 (.node 19949 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5651_ordered : SortedNatTree.ordered row5651 = true := by decide
set_option maxRecDepth 4096 in
theorem row5651_values : SortedNatTree.values row5651 = Data.row5651 := rfl
theorem row5651_checked : SortedNatTree.contains row5651 = Data.row5651.contains := by
  rw [SortedNatTree.contains_eq_list row5651 row5651_ordered, row5651_values]
def row5653 : SortedNatTree.Tree := (.node 10357 (.node 7333 (.node 6211 (.node 6079 (.node 5923 (.node 5827 .empty .empty) .empty) (.node 6133 .empty .empty)) (.node 6619 (.node 6367 (.node 6247 .empty .empty) .empty) (.node 6793 .empty .empty))) (.node 8563 (.node 8353 (.node 8269 (.node 7549 .empty .empty) .empty) (.node 8431 .empty .empty)) (.node 9013 (.node 8929 (.node 8731 .empty .empty) .empty) (.node 9601 .empty .empty)))) (.node 15991 (.node 12919 (.node 11083 (.node 10987 (.node 10939 .empty .empty) .empty) (.node 11119 .empty .empty)) (.node 14173 (.node 13921 (.node 13591 .empty .empty) .empty) (.node 15187 .empty .empty))) (.node 17839 (.node 17539 (.node 17497 (.node 17047 .empty .empty) .empty) (.node 17761 .empty .empty)) (.node 18211 (.node 18043 .empty .empty) (.node 20149 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5653_ordered : SortedNatTree.ordered row5653 = true := by decide
set_option maxRecDepth 4096 in
theorem row5653_values : SortedNatTree.values row5653 = Data.row5653 := rfl
theorem row5653_checked : SortedNatTree.contains row5653 = Data.row5653.contains := by
  rw [SortedNatTree.contains_eq_list row5653 row5653_ordered, row5653_values]
def row5657 : SortedNatTree.Tree := (.node 11813 (.node 8369 (.node 7673 (.node 6983 (.node 6791 .empty .empty) (.node 7529 .empty .empty)) (.node 8009 (.node 7823 .empty .empty) .empty)) (.node 10781 (.node 9923 (.node 8387 .empty .empty) (.node 10259 .empty .empty)) (.node 11717 (.node 10979 .empty .empty) .empty))) (.node 16103 (.node 13751 (.node 12491 (.node 11897 .empty .empty) (.node 13649 .empty .empty)) (.node 15809 (.node 14783 .empty .empty) .empty)) (.node 18119 (.node 17387 (.node 16691 .empty .empty) (.node 18077 .empty .empty)) (.node 18371 (.node 18269 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5657_ordered : SortedNatTree.ordered row5657 = true := by decide
set_option maxRecDepth 4096 in
theorem row5657_values : SortedNatTree.values row5657 = Data.row5657 := rfl
theorem row5657_checked : SortedNatTree.contains row5657 = Data.row5657.contains := by
  rw [SortedNatTree.contains_eq_list row5657 row5657_ordered, row5657_values]
def row5659 : SortedNatTree.Tree := (.node 11587 (.node 7759 (.node 6271 (.node 5953 (.node 5779 (.node 5701 .empty .empty) .empty) (.node 6151 (.node 6121 .empty .empty) .empty)) (.node 6703 (.node 6619 (.node 6577 .empty .empty) .empty) (.node 7069 .empty .empty))) (.node 8677 (.node 7993 (.node 7927 (.node 7867 .empty .empty) .empty) (.node 8461 .empty .empty)) (.node 10993 (.node 10567 (.node 10069 .empty .empty) .empty) (.node 11329 .empty .empty)))) (.node 16657 (.node 14551 (.node 13093 (.node 12421 (.node 11779 .empty .empty) .empty) (.node 14347 .empty .empty)) (.node 15607 (.node 14851 (.node 14653 .empty .empty) .empty) (.node 16069 .empty .empty))) (.node 19231 (.node 18583 (.node 18043 (.node 17467 .empty .empty) .empty) (.node 18757 .empty .empty)) (.node 19687 (.node 19609 (.node 19429 .empty .empty) .empty) (.node 20269 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5659_ordered : SortedNatTree.ordered row5659 = true := by decide
set_option maxRecDepth 4096 in
theorem row5659_values : SortedNatTree.values row5659 = Data.row5659 := rfl
theorem row5659_checked : SortedNatTree.contains row5659 = Data.row5659.contains := by
  rw [SortedNatTree.contains_eq_list row5659 row5659_ordered, row5659_values]
def row5669 : SortedNatTree.Tree := (.node 15497 (.node 8819 (.node 7559 (.node 6011 (.node 5879 .empty .empty) (.node 7517 .empty .empty)) (.node 8627 (.node 8573 .empty .empty) (.node 8693 .empty .empty))) (.node 12251 (.node 9491 (.node 9479 .empty .empty) (.node 10037 .empty .empty)) (.node 13829 (.node 12743 .empty .empty) (.node 15263 .empty .empty)))) (.node 17909 (.node 16451 (.node 16253 (.node 15641 .empty .empty) (.node 16421 .empty .empty)) (.node 17093 (.node 16937 .empty .empty) (.node 17903 .empty .empty))) (.node 19559 (.node 19373 (.node 19139 .empty .empty) (.node 19541 .empty .empty)) (.node 20147 (.node 20051 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5669_ordered : SortedNatTree.ordered row5669 = true := by decide
set_option maxRecDepth 4096 in
theorem row5669_values : SortedNatTree.values row5669 = Data.row5669 := rfl
theorem row5669_checked : SortedNatTree.contains row5669 = Data.row5669.contains := by
  rw [SortedNatTree.contains_eq_list row5669 row5669_ordered, row5669_values]
def row5683 : SortedNatTree.Tree := (.node 14221 (.node 7369 (.node 6907 (.node 6271 (.node 6217 .empty .empty) .empty) (.node 7159 (.node 7057 .empty .empty) .empty)) (.node 9661 (.node 9103 (.node 9091 .empty .empty) .empty) (.node 13063 (.node 12541 .empty .empty) .empty))) (.node 17839 (.node 15823 (.node 15739 (.node 14683 .empty .empty) .empty) (.node 17707 (.node 16267 .empty .empty) .empty)) (.node 20023 (.node 18301 (.node 18199 .empty .empty) .empty) (.node 20161 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5683_ordered : SortedNatTree.ordered row5683 = true := by decide
set_option maxRecDepth 4096 in
theorem row5683_values : SortedNatTree.values row5683 = Data.row5683 := rfl
theorem row5683_checked : SortedNatTree.contains row5683 = Data.row5683.contains := by
  rw [SortedNatTree.contains_eq_list row5683 row5683_ordered, row5683_values]
def row5689 : SortedNatTree.Tree := (.node 11257 (.node 7297 (.node 6271 (.node 6163 (.node 5749 .empty .empty) (.node 6211 .empty .empty)) (.node 6673 (.node 6277 .empty .empty) (.node 6823 .empty .empty))) (.node 8689 (.node 8329 (.node 7573 .empty .empty) (.node 8419 .empty .empty)) (.node 9679 (.node 9127 .empty .empty) (.node 10099 .empty .empty)))) (.node 15643 (.node 15121 (.node 12451 (.node 11503 .empty .empty) (.node 15031 .empty .empty)) (.node 15349 (.node 15199 .empty .empty) (.node 15601 .empty .empty))) (.node 19231 (.node 18367 (.node 17467 .empty .empty) (.node 18457 .empty .empty)) (.node 19333 (.node 19249 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5689_ordered : SortedNatTree.ordered row5689 = true := by decide
set_option maxRecDepth 4096 in
theorem row5689_values : SortedNatTree.values row5689 = Data.row5689 := rfl
theorem row5689_checked : SortedNatTree.contains row5689 = Data.row5689.contains := by
  rw [SortedNatTree.contains_eq_list row5689 row5689_ordered, row5689_values]
def row5693 : SortedNatTree.Tree := (.node 13553 (.node 8741 (.node 7187 (.node 6563 (.node 6329 .empty .empty) (.node 7103 .empty .empty)) (.node 8699 (.node 8171 .empty .empty) .empty)) (.node 11411 (.node 10211 (.node 9689 .empty .empty) (.node 11177 .empty .empty)) (.node 12119 (.node 12041 .empty .empty) .empty))) (.node 15629 (.node 14771 (.node 14033 (.node 13781 .empty .empty) (.node 14519 .empty .empty)) (.node 15287 (.node 15077 .empty .empty) .empty)) (.node 18041 (.node 17393 (.node 15731 .empty .empty) (.node 17681 .empty .empty)) (.node 19979 (.node 19913 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5693_ordered : SortedNatTree.ordered row5693 = true := by decide
set_option maxRecDepth 4096 in
theorem row5693_values : SortedNatTree.values row5693 = Data.row5693 := rfl
theorem row5693_checked : SortedNatTree.contains row5693 = Data.row5693.contains := by
  rw [SortedNatTree.contains_eq_list row5693 row5693_ordered, row5693_values]
def row5701 : SortedNatTree.Tree := (.node 11959 (.node 8209 (.node 6733 (.node 6481 (.node 6367 .empty .empty) .empty) (.node 7669 (.node 7459 .empty .empty) .empty)) (.node 9439 (.node 8707 (.node 8389 .empty .empty) .empty) (.node 10939 .empty .empty))) (.node 14983 (.node 14851 (.node 14221 (.node 13477 .empty .empty) .empty) (.node 14947 .empty .empty)) (.node 16903 (.node 15607 (.node 15187 .empty .empty) .empty) (.node 17203 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5701_ordered : SortedNatTree.ordered row5701 = true := by decide
set_option maxRecDepth 4096 in
theorem row5701_values : SortedNatTree.values row5701 = Data.row5701 := rfl
theorem row5701_checked : SortedNatTree.contains row5701 = Data.row5701.contains := by
  rw [SortedNatTree.contains_eq_list row5701 row5701_ordered, row5701_values]
def row5711 : SortedNatTree.Tree := (.node 11003 (.node 7523 (.node 6173 (.node 5867 (.node 5783 .empty .empty) (.node 5939 .empty .empty)) (.node 7103 (.node 7013 .empty .empty) (.node 7451 .empty .empty))) (.node 8291 (.node 8069 (.node 7883 .empty .empty) (.node 8273 .empty .empty)) (.node 9341 (.node 8573 .empty .empty) (.node 10781 .empty .empty)))) (.node 14717 (.node 12239 (.node 11789 (.node 11351 .empty .empty) (.node 11903 .empty .empty)) (.node 13859 (.node 13619 .empty .empty) (.node 14669 .empty .empty))) (.node 16103 (.node 15131 (.node 14759 .empty .empty) (.node 15401 .empty .empty)) (.node 18059 (.node 17231 .empty .empty) (.node 18803 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5711_ordered : SortedNatTree.ordered row5711 = true := by decide
set_option maxRecDepth 4096 in
theorem row5711_values : SortedNatTree.values row5711 = Data.row5711 := rfl
theorem row5711_checked : SortedNatTree.contains row5711 = Data.row5711.contains := by
  rw [SortedNatTree.contains_eq_list row5711 row5711_ordered, row5711_values]
def row5717 : SortedNatTree.Tree := (.node 13397 (.node 10487 (.node 9371 (.node 8171 (.node 7079 .empty .empty) .empty) (.node 9887 (.node 9431 .empty .empty) .empty)) (.node 11783 (.node 11633 (.node 10631 .empty .empty) .empty) (.node 12347 .empty .empty))) (.node 16763 (.node 15461 (.node 15233 (.node 14159 .empty .empty) .empty) (.node 16529 (.node 16073 .empty .empty) .empty)) (.node 17891 (.node 17789 (.node 17117 .empty .empty) .empty) (.node 18401 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5717_ordered : SortedNatTree.ordered row5717 = true := by decide
set_option maxRecDepth 4096 in
theorem row5717_values : SortedNatTree.values row5717 = Data.row5717 := rfl
theorem row5717_checked : SortedNatTree.contains row5717 = Data.row5717.contains := by
  rw [SortedNatTree.contains_eq_list row5717 row5717_ordered, row5717_values]
def row5737 : SortedNatTree.Tree := (.node 10987 (.node 8779 (.node 8263 (.node 7213 (.node 6301 .empty .empty) (.node 7687 .empty .empty)) (.node 8677 (.node 8521 .empty .empty) .empty)) (.node 9769 (.node 8941 (.node 8929 .empty .empty) (.node 9613 .empty .empty)) (.node 10957 (.node 10069 .empty .empty) .empty))) (.node 16699 (.node 14389 (.node 13171 (.node 12511 .empty .empty) (.node 13597 .empty .empty)) (.node 15739 (.node 14593 .empty .empty) .empty)) (.node 18181 (.node 17791 (.node 17029 .empty .empty) (.node 17929 .empty .empty)) (.node 19183 (.node 19051 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5737_ordered : SortedNatTree.ordered row5737 = true := by decide
set_option maxRecDepth 4096 in
theorem row5737_values : SortedNatTree.values row5737 = Data.row5737 := rfl
theorem row5737_checked : SortedNatTree.contains row5737 = Data.row5737.contains := by
  rw [SortedNatTree.contains_eq_list row5737 row5737_ordered, row5737_values]
def row5741 : SortedNatTree.Tree := (.node 12809 (.node 10091 (.node 7541 (.node 6833 (.node 6299 .empty .empty) (.node 6959 .empty .empty)) (.node 8219 (.node 8093 .empty .empty) .empty)) (.node 11633 (.node 11519 (.node 10343 .empty .empty) (.node 11621 .empty .empty)) (.node 12323 (.node 11987 .empty .empty) .empty))) (.node 17681 (.node 15383 (.node 14153 (.node 13931 .empty .empty) (.node 15173 .empty .empty)) (.node 17207 (.node 15803 .empty .empty) .empty)) (.node 19289 (.node 17987 (.node 17807 .empty .empty) (.node 18143 .empty .empty)) (.node 19841 (.node 19379 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5741_ordered : SortedNatTree.ordered row5741 = true := by decide
set_option maxRecDepth 4096 in
theorem row5741_values : SortedNatTree.values row5741 = Data.row5741 := rfl
theorem row5741_checked : SortedNatTree.contains row5741 = Data.row5741.contains := by
  rw [SortedNatTree.contains_eq_list row5741 row5741_ordered, row5741_values]
def row5743 : SortedNatTree.Tree := (.node 10159 (.node 7717 (.node 7507 (.node 6271 (.node 5857 .empty .empty) (.node 7027 .empty .empty)) (.node 7591 (.node 7573 .empty .empty) .empty)) (.node 9043 (.node 8287 (.node 7759 .empty .empty) (.node 8377 .empty .empty)) (.node 10111 (.node 9883 .empty .empty) .empty))) (.node 14107 (.node 12799 (.node 10909 (.node 10369 .empty .empty) (.node 12049 .empty .empty)) (.node 13921 (.node 13591 .empty .empty) .empty)) (.node 16747 (.node 15787 (.node 14551 .empty .empty) (.node 16651 .empty .empty)) (.node 18289 (.node 17713 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5743_ordered : SortedNatTree.ordered row5743 = true := by decide
set_option maxRecDepth 4096 in
theorem row5743_values : SortedNatTree.values row5743 = Data.row5743 := rfl
theorem row5743_checked : SortedNatTree.contains row5743 = Data.row5743.contains := by
  rw [SortedNatTree.contains_eq_list row5743 row5743_ordered, row5743_values]
def row5749 : SortedNatTree.Tree := (.node 12433 (.node 8929 (.node 6427 (.node 6199 (.node 6007 (.node 5839 .empty .empty) .empty) (.node 6211 .empty .empty)) (.node 7753 (.node 7681 (.node 7537 .empty .empty) .empty) (.node 8101 .empty .empty))) (.node 10459 (.node 9907 (.node 9829 (.node 9109 .empty .empty) .empty) (.node 10099 .empty .empty)) (.node 11131 (.node 10723 (.node 10513 .empty .empty) .empty) (.node 12277 .empty .empty)))) (.node 16987 (.node 13873 (.node 13249 (.node 12781 (.node 12487 .empty .empty) .empty) (.node 13291 .empty .empty)) (.node 16057 (.node 15973 (.node 15307 .empty .empty) .empty) (.node 16921 .empty .empty))) (.node 19231 (.node 18061 (.node 17389 (.node 17053 .empty .empty) .empty) (.node 18181 .empty .empty)) (.node 19963 (.node 19333 .empty .empty) (.node 20113 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5749_ordered : SortedNatTree.ordered row5749 = true := by decide
set_option maxRecDepth 4096 in
theorem row5749_values : SortedNatTree.values row5749 = Data.row5749 := rfl
theorem row5749_checked : SortedNatTree.contains row5749 = Data.row5749.contains := by
  rw [SortedNatTree.contains_eq_list row5749 row5749_ordered, row5749_values]
def row5779 : SortedNatTree.Tree := (.node 10477 (.node 8677 (.node 7717 (.node 6421 (.node 6007 .empty .empty) (.node 7591 .empty .empty)) (.node 8443 (.node 8269 .empty .empty) .empty)) (.node 9109 (.node 8761 (.node 8731 .empty .empty) (.node 8941 .empty .empty)) (.node 9769 (.node 9619 .empty .empty) .empty))) (.node 17551 (.node 14737 (.node 14011 (.node 11119 .empty .empty) (.node 14347 .empty .empty)) (.node 15607 (.node 15601 .empty .empty) .empty)) (.node 18379 (.node 17749 (.node 17707 .empty .empty) (.node 17881 .empty .empty)) (.node 19891 (.node 19387 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5779_ordered : SortedNatTree.ordered row5779 = true := by decide
set_option maxRecDepth 4096 in
theorem row5779_values : SortedNatTree.values row5779 = Data.row5779 := rfl
theorem row5779_checked : SortedNatTree.contains row5779 = Data.row5779.contains := by
  rw [SortedNatTree.contains_eq_list row5779 row5779_ordered, row5779_values]
def row5783 : SortedNatTree.Tree := (.node 11897 (.node 9239 (.node 6983 (.node 6719 (.node 6491 .empty .empty) (.node 6959 .empty .empty)) (.node 8009 (.node 7331 .empty .empty) (.node 9011 .empty .empty))) (.node 10151 (.node 9851 (.node 9689 .empty .empty) (.node 10133 .empty .empty)) (.node 11717 (.node 11087 .empty .empty) (.node 11777 .empty .empty)))) (.node 16787 (.node 13721 (.node 13217 (.node 12659 .empty .empty) (.node 13691 .empty .empty)) (.node 15131 (.node 13763 .empty .empty) (.node 15443 .empty .empty))) (.node 18119 (.node 17903 (.node 17123 .empty .empty) (.node 18077 .empty .empty)) (.node 20147 (.node 18269 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5783_ordered : SortedNatTree.ordered row5783 = true := by decide
set_option maxRecDepth 4096 in
theorem row5783_values : SortedNatTree.values row5783 = Data.row5783 := rfl
theorem row5783_checked : SortedNatTree.contains row5783 = Data.row5783.contains := by
  rw [SortedNatTree.contains_eq_list row5783 row5783_ordered, row5783_values]
def row5791 : SortedNatTree.Tree := (.node 13693 (.node 9349 (.node 8317 (.node 7333 (.node 5839 .empty .empty) (.node 8293 .empty .empty)) (.node 9091 (.node 8467 .empty .empty) .empty)) (.node 10501 (.node 10243 (.node 9601 .empty .empty) .empty) (.node 12613 (.node 11551 .empty .empty) .empty))) (.node 18097 (.node 16189 (.node 14629 (.node 14461 .empty .empty) (.node 15451 .empty .empty)) (.node 17569 (.node 16843 .empty .empty) .empty)) (.node 19249 (.node 19207 (.node 18517 .empty .empty) .empty) (.node 20089 (.node 19423 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5791_ordered : SortedNatTree.ordered row5791 = true := by decide
set_option maxRecDepth 4096 in
theorem row5791_values : SortedNatTree.values row5791 = Data.row5791 := rfl
theorem row5791_checked : SortedNatTree.contains row5791 = Data.row5791.contains := by
  rw [SortedNatTree.contains_eq_list row5791 row5791_ordered, row5791_values]
def row5801 : SortedNatTree.Tree := (.node 11027 (.node 8693 (.node 6173 (.node 6029 (.node 5897 .empty .empty) (.node 6131 .empty .empty)) (.node 7919 (.node 6263 .empty .empty) .empty)) (.node 9491 (.node 8963 (.node 8783 .empty .empty) (.node 9371 .empty .empty)) (.node 11003 (.node 10253 .empty .empty) .empty))) (.node 15461 (.node 12983 (.node 11261 (.node 11243 .empty .empty) (.node 11471 .empty .empty)) (.node 14621 (.node 13841 .empty .empty) .empty)) (.node 18191 (.node 17183 (.node 17093 .empty .empty) (.node 17789 .empty .empty)) (.node 19013 (.node 18899 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5801_ordered : SortedNatTree.ordered row5801 = true := by decide
set_option maxRecDepth 4096 in
theorem row5801_values : SortedNatTree.values row5801 = Data.row5801 := rfl
theorem row5801_checked : SortedNatTree.contains row5801 = Data.row5801.contains := by
  rw [SortedNatTree.contains_eq_list row5801 row5801_ordered, row5801_values]
def row5807 : SortedNatTree.Tree := (.node 10799 (.node 9221 (.node 7949 (.node 7499 (.node 6329 .empty .empty) (.node 7577 .empty .empty)) (.node 9059 (.node 8297 .empty .empty) .empty)) (.node 9587 (.node 9551 (.node 9473 .empty .empty) .empty) (.node 10211 (.node 9689 .empty .empty) .empty))) (.node 16529 (.node 13781 (.node 11321 (.node 10853 .empty .empty) (.node 13451 .empty .empty)) (.node 15101 (.node 14087 .empty .empty) .empty)) (.node 18233 (.node 17789 (.node 16823 .empty .empty) .empty) (.node 19583 (.node 18539 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5807_ordered : SortedNatTree.ordered row5807 = true := by decide
set_option maxRecDepth 4096 in
theorem row5807_values : SortedNatTree.values row5807 = Data.row5807 := rfl
theorem row5807_checked : SortedNatTree.contains row5807 = Data.row5807.contains := by
  rw [SortedNatTree.contains_eq_list row5807 row5807_ordered, row5807_values]
def row5813 : SortedNatTree.Tree := (.node 10139 (.node 7907 (.node 7151 (.node 6803 (.node 6737 .empty .empty) (.node 6863 .empty .empty)) (.node 7517 (.node 7193 .empty .empty) (.node 7829 .empty .empty))) (.node 8867 (.node 8669 (.node 8363 .empty .empty) (.node 8831 .empty .empty)) (.node 10091 (.node 10007 .empty .empty) .empty))) (.node 16607 (.node 13751 (.node 12959 (.node 12659 .empty .empty) (.node 13451 .empty .empty)) (.node 14957 (.node 14639 .empty .empty) (.node 16481 .empty .empty))) (.node 18671 (.node 18371 (.node 17747 .empty .empty) (.node 18539 .empty .empty)) (.node 19163 (.node 18797 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5813_ordered : SortedNatTree.ordered row5813 = true := by decide
set_option maxRecDepth 4096 in
theorem row5813_values : SortedNatTree.values row5813 = Data.row5813 := rfl
theorem row5813_checked : SortedNatTree.contains row5813 = Data.row5813.contains := by
  rw [SortedNatTree.contains_eq_list row5813 row5813_ordered, row5813_values]
def row5821 : SortedNatTree.Tree := (.node 12721 (.node 8233 (.node 6637 (.node 6121 (.node 6043 .empty .empty) (.node 6373 .empty .empty)) (.node 8059 (.node 6967 .empty .empty) .empty)) (.node 10663 (.node 9697 (.node 9397 .empty .empty) (.node 10567 .empty .empty)) (.node 11923 (.node 11443 .empty .empty) .empty))) (.node 16903 (.node 14389 (.node 13291 (.node 12841 .empty .empty) (.node 14341 .empty .empty)) (.node 14737 (.node 14479 .empty .empty) .empty)) (.node 18211 (.node 17047 (.node 16987 .empty .empty) .empty) (.node 19231 (.node 19183 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5821_ordered : SortedNatTree.ordered row5821 = true := by decide
set_option maxRecDepth 4096 in
theorem row5821_values : SortedNatTree.values row5821 = Data.row5821 := rfl
theorem row5821_checked : SortedNatTree.contains row5821 = Data.row5821.contains := by
  rw [SortedNatTree.contains_eq_list row5821 row5821_ordered, row5821_values]
def row5827 : SortedNatTree.Tree := (.node 12073 (.node 8053 (.node 7213 (.node 6991 (.node 6151 .empty .empty) (.node 7159 .empty .empty)) (.node 7621 (.node 7507 .empty .empty) .empty)) (.node 9547 (.node 9103 (.node 8089 .empty .empty) (.node 9127 .empty .empty)) (.node 9679 (.node 9601 .empty .empty) .empty))) (.node 14779 (.node 14437 (.node 13723 (.node 12511 .empty .empty) (.node 13933 .empty .empty)) (.node 14563 (.node 14557 .empty .empty) .empty)) (.node 18541 (.node 17929 (.node 16363 .empty .empty) .empty) (.node 20011 (.node 19717 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5827_ordered : SortedNatTree.ordered row5827 = true := by decide
set_option maxRecDepth 4096 in
theorem row5827_values : SortedNatTree.values row5827 = Data.row5827 := rfl
theorem row5827_checked : SortedNatTree.contains row5827 = Data.row5827.contains := by
  rw [SortedNatTree.contains_eq_list row5827 row5827_ordered, row5827_values]
def row5839 : SortedNatTree.Tree := (.node 10597 (.node 8521 (.node 7753 (.node 7237 (.node 6733 (.node 6199 .empty .empty) .empty) (.node 7681 .empty .empty)) (.node 8233 (.node 7879 .empty .empty) (.node 8269 .empty .empty))) (.node 9619 (.node 9013 (.node 8539 .empty .empty) (.node 9391 .empty .empty)) (.node 10333 (.node 9907 .empty .empty) (.node 10501 .empty .empty)))) (.node 15331 (.node 13633 (.node 11701 (.node 11047 .empty .empty) (.node 13441 .empty .empty)) (.node 15121 (.node 13999 .empty .empty) (.node 15187 .empty .empty))) (.node 18913 (.node 15823 (.node 15583 .empty .empty) (.node 17881 .empty .empty)) (.node 19753 (.node 19081 .empty .empty) (.node 20107 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row5839_ordered : SortedNatTree.ordered row5839 = true := by decide
set_option maxRecDepth 4096 in
theorem row5839_values : SortedNatTree.values row5839 = Data.row5839 := rfl
theorem row5839_checked : SortedNatTree.contains row5839 = Data.row5839.contains := by
  rw [SortedNatTree.contains_eq_list row5839 row5839_ordered, row5839_values]
def row5843 : SortedNatTree.Tree := (.node 13523 (.node 9749 (.node 7937 (.node 7583 (.node 6977 .empty .empty) .empty) (.node 9533 (.node 8693 .empty .empty) .empty)) (.node 11549 (.node 11171 (.node 9767 .empty .empty) .empty) (.node 13421 (.node 11657 .empty .empty) .empty))) (.node 19157 (.node 16451 (.node 16217 (.node 14867 .empty .empty) .empty) (.node 18131 (.node 16661 .empty .empty) .empty)) (.node 19763 (.node 19661 (.node 19553 .empty .empty) .empty) (.node 20063 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5843_ordered : SortedNatTree.ordered row5843 = true := by decide
set_option maxRecDepth 4096 in
theorem row5843_values : SortedNatTree.values row5843 = Data.row5843 := rfl
theorem row5843_checked : SortedNatTree.contains row5843 = Data.row5843.contains := by
  rw [SortedNatTree.contains_eq_list row5843 row5843_ordered, row5843_values]
def row5849 : SortedNatTree.Tree := (.node 11351 (.node 9929 (.node 8681 (.node 7949 (.node 6203 .empty .empty) .empty) (.node 8753 .empty .empty)) (.node 10691 (.node 10427 (.node 10259 .empty .empty) .empty) (.node 10979 .empty .empty))) (.node 15173 (.node 12821 (.node 12527 (.node 12161 .empty .empty) .empty) (.node 13457 .empty .empty)) (.node 17981 (.node 16931 (.node 15767 .empty .empty) .empty) (.node 18869 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row5849_ordered : SortedNatTree.ordered row5849 = true := by decide
set_option maxRecDepth 4096 in
theorem row5849_values : SortedNatTree.values row5849 = Data.row5849 := rfl
theorem row5849_checked : SortedNatTree.contains row5849 = Data.row5849.contains := by
  rw [SortedNatTree.contains_eq_list row5849 row5849_ordered, row5849_values]
def row5851 : SortedNatTree.Tree := (.node 13063 (.node 9463 (.node 6961 (.node 6607 (.node 6373 .empty .empty) (.node 6793 .empty .empty)) (.node 8377 (.node 7507 .empty .empty) .empty)) (.node 11257 (.node 10837 (.node 9733 .empty .empty) .empty) (.node 12097 (.node 11959 .empty .empty) .empty))) (.node 18379 (.node 15439 (.node 14419 (.node 13567 .empty .empty) .empty) (.node 16843 (.node 15649 .empty .empty) .empty)) (.node 19597 (.node 18637 (.node 18493 .empty .empty) .empty) (.node 20149 (.node 19891 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5851_ordered : SortedNatTree.ordered row5851 = true := by decide
set_option maxRecDepth 4096 in
theorem row5851_values : SortedNatTree.values row5851 = Data.row5851 := rfl
theorem row5851_checked : SortedNatTree.contains row5851 = Data.row5851.contains := by
  rw [SortedNatTree.contains_eq_list row5851 row5851_ordered, row5851_values]
def row5857 : SortedNatTree.Tree := (.node 13729 (.node 9463 (.node 8059 (.node 6481 (.node 5953 .empty .empty) (.node 7723 .empty .empty)) (.node 9007 (.node 8293 .empty .empty) .empty)) (.node 12277 (.node 11839 (.node 9511 .empty .empty) .empty) (.node 13417 (.node 12763 .empty .empty) .empty))) (.node 16987 (.node 15583 (.node 14461 (.node 14149 .empty .empty) .empty) (.node 16633 (.node 16381 .empty .empty) .empty)) (.node 19237 (.node 18229 (.node 17401 .empty .empty) .empty) (.node 19843 (.node 19273 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row5857_ordered : SortedNatTree.ordered row5857 = true := by decide
set_option maxRecDepth 4096 in
theorem row5857_values : SortedNatTree.values row5857 = Data.row5857 := rfl
theorem row5857_checked : SortedNatTree.contains row5857 = Data.row5857.contains := by
  rw [SortedNatTree.contains_eq_list row5857 row5857_ordered, row5857_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
