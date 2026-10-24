import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row2671 : SortedNatTree.Tree := (.node 11617 (.node 6829 (.node 4603 (.node 3769 (.node 3571 (.node 3361 .empty .empty) .empty) (.node 4231 (.node 3853 .empty .empty) .empty)) (.node 6343 (.node 6211 (.node 5077 .empty .empty) .empty) (.node 6709 .empty .empty))) (.node 9067 (.node 7723 (.node 7507 (.node 7129 .empty .empty) .empty) (.node 8761 (.node 8053 .empty .empty) .empty)) (.node 10651 (.node 10159 (.node 9811 .empty .empty) .empty) (.node 11287 .empty .empty)))) (.node 18541 (.node 15907 (.node 13633 (.node 13177 (.node 12889 .empty .empty) .empty) (.node 14983 (.node 14437 .empty .empty) .empty)) (.node 18307 (.node 17971 (.node 17623 .empty .empty) .empty) (.node 18457 .empty .empty))) (.node 20611 (.node 19843 (.node 19597 (.node 18583 .empty .empty) .empty) (.node 20233 .empty .empty)) (.node 21319 (.node 21193 (.node 21121 .empty .empty) .empty) (.node 22993 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2671_ordered : SortedNatTree.ordered row2671 = true := by decide
set_option maxRecDepth 4096 in
theorem row2671_values : SortedNatTree.values row2671 = Data.row2671 := rfl
theorem row2671_checked : SortedNatTree.contains row2671 = Data.row2671.contains := by
  rw [SortedNatTree.contains_eq_list row2671 row2671_ordered, row2671_values]
def row2677 : SortedNatTree.Tree := (.node 9859 (.node 5851 (.node 4261 (.node 3049 (.node 2971 (.node 2833 .empty .empty) .empty) (.node 3469 .empty .empty)) (.node 5569 (.node 5563 .empty .empty) (.node 5737 .empty .empty))) (.node 8419 (.node 7369 (.node 6961 (.node 6247 .empty .empty) .empty) (.node 7741 .empty .empty)) (.node 9007 (.node 8707 .empty .empty) (.node 9043 .empty .empty)))) (.node 17929 (.node 13879 (.node 13177 (.node 12007 (.node 11701 .empty .empty) .empty) (.node 13291 .empty .empty)) (.node 15493 (.node 14767 .empty .empty) (.node 17623 .empty .empty))) (.node 20443 (.node 18097 (.node 18013 .empty .empty) (.node 19819 .empty .empty)) (.node 21613 (.node 21031 .empty .empty) (.node 22153 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2677_ordered : SortedNatTree.ordered row2677 = true := by decide
set_option maxRecDepth 4096 in
theorem row2677_values : SortedNatTree.values row2677 = Data.row2677 := rfl
theorem row2677_checked : SortedNatTree.contains row2677 = Data.row2677.contains := by
  rw [SortedNatTree.contains_eq_list row2677 row2677_ordered, row2677_values]
def row2683 : SortedNatTree.Tree := (.node 11149 (.node 6451 (.node 3931 (.node 3253 (.node 2797 (.node 2791 .empty .empty) .empty) (.node 3823 .empty .empty)) (.node 5209 (.node 5023 (.node 4447 .empty .empty) .empty) (.node 5563 .empty .empty))) (.node 8803 (.node 8167 (.node 8059 (.node 6619 .empty .empty) .empty) (.node 8293 .empty .empty)) (.node 10831 (.node 10687 (.node 9067 .empty .empty) .empty) (.node 11047 .empty .empty)))) (.node 16141 (.node 12973 (.node 11821 (.node 11689 (.node 11467 .empty .empty) .empty) (.node 12601 .empty .empty)) (.node 14887 (.node 14419 (.node 13411 .empty .empty) .empty) (.node 15349 .empty .empty))) (.node 22189 (.node 17659 (.node 17551 (.node 17317 .empty .empty) .empty) (.node 19489 .empty .empty)) (.node 22651 (.node 22447 .empty .empty) (.node 22777 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2683_ordered : SortedNatTree.ordered row2683 = true := by decide
set_option maxRecDepth 4096 in
theorem row2683_values : SortedNatTree.values row2683 = Data.row2683 := rfl
theorem row2683_checked : SortedNatTree.contains row2683 = Data.row2683.contains := by
  rw [SortedNatTree.contains_eq_list row2683 row2683_ordered, row2683_values]
def row2687 : SortedNatTree.Tree := (.node 11213 (.node 8069 (.node 6221 (.node 4481 (.node 4007 (.node 3023 .empty .empty) .empty) (.node 5897 (.node 5237 .empty .empty) .empty)) (.node 7703 (.node 6737 (.node 6599 .empty .empty) .empty) (.node 8039 .empty .empty))) (.node 8693 (.node 8543 (.node 8363 (.node 8231 .empty .empty) .empty) (.node 8627 (.node 8573 .empty .empty) .empty)) (.node 10457 (.node 10193 (.node 9551 .empty .empty) .empty) (.node 11087 .empty .empty)))) (.node 19001 (.node 16223 (.node 14447 (.node 13469 (.node 13121 .empty .empty) .empty) (.node 15971 (.node 14813 .empty .empty) .empty)) (.node 17291 (.node 16883 (.node 16493 .empty .empty) .empty) (.node 17783 .empty .empty))) (.node 21767 (.node 20117 (.node 19961 (.node 19421 .empty .empty) .empty) (.node 20261 .empty .empty)) (.node 22907 (.node 22697 (.node 21773 .empty .empty) .empty) (.node 23201 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2687_ordered : SortedNatTree.ordered row2687 = true := by decide
set_option maxRecDepth 4096 in
theorem row2687_values : SortedNatTree.values row2687 = Data.row2687 := rfl
theorem row2687_checked : SortedNatTree.contains row2687 = Data.row2687.contains := by
  rw [SortedNatTree.contains_eq_list row2687 row2687_ordered, row2687_values]
def row2689 : SortedNatTree.Tree := (.node 10009 (.node 6763 (.node 5347 (.node 4423 (.node 3343 (.node 3271 .empty .empty) .empty) (.node 4507 .empty .empty)) (.node 6379 (.node 6367 (.node 6043 .empty .empty) .empty) (.node 6679 .empty .empty))) (.node 8317 (.node 8161 (.node 8089 (.node 7477 .empty .empty) .empty) (.node 8191 .empty .empty)) (.node 9013 (.node 8761 (.node 8527 .empty .empty) .empty) (.node 9643 .empty .empty)))) (.node 19009 (.node 15919 (.node 12907 (.node 12697 (.node 10891 .empty .empty) .empty) (.node 14731 .empty .empty)) (.node 17989 (.node 17041 (.node 16381 .empty .empty) .empty) (.node 18379 .empty .empty))) (.node 21787 (.node 21319 (.node 20593 (.node 19309 .empty .empty) .empty) (.node 21391 .empty .empty)) (.node 22273 (.node 22189 .empty .empty) (.node 22921 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2689_ordered : SortedNatTree.ordered row2689 = true := by decide
set_option maxRecDepth 4096 in
theorem row2689_values : SortedNatTree.values row2689 = Data.row2689 := rfl
theorem row2689_checked : SortedNatTree.contains row2689 = Data.row2689.contains := by
  rw [SortedNatTree.contains_eq_list row2689 row2689_ordered, row2689_values]
def row2693 : SortedNatTree.Tree := (.node 12161 (.node 6779 (.node 5879 (.node 4373 (.node 4079 (.node 4049 .empty .empty) (.node 4139 .empty .empty)) (.node 5651 (.node 4931 .empty .empty) .empty)) (.node 6113 (.node 6011 (.node 5903 .empty .empty) .empty) (.node 6263 (.node 6257 .empty .empty) .empty))) (.node 8501 (.node 7331 (.node 7103 (.node 7013 .empty .empty) .empty) (.node 8123 (.node 7901 .empty .empty) .empty)) (.node 9533 (.node 9431 (.node 9239 .empty .empty) .empty) (.node 11177 (.node 9833 .empty .empty) .empty)))) (.node 15797 (.node 13217 (.node 12713 (.node 12671 (.node 12269 .empty .empty) .empty) (.node 12893 (.node 12809 .empty .empty) .empty)) (.node 14327 (.node 13907 (.node 13553 .empty .empty) .empty) (.node 15161 (.node 15077 .empty .empty) .empty))) (.node 18443 (.node 17681 (.node 17099 (.node 16319 .empty .empty) .empty) (.node 18341 (.node 18089 .empty .empty) .empty)) (.node 20393 (.node 19991 (.node 18593 .empty .empty) .empty) (.node 23039 (.node 21863 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2693_ordered : SortedNatTree.ordered row2693 = true := by decide
set_option maxRecDepth 4096 in
theorem row2693_values : SortedNatTree.values row2693 = Data.row2693 := rfl
theorem row2693_checked : SortedNatTree.contains row2693 = Data.row2693.contains := by
  rw [SortedNatTree.contains_eq_list row2693 row2693_ordered, row2693_values]
def row2699 : SortedNatTree.Tree := (.node 12611 (.node 7121 (.node 4133 (.node 3413 (.node 3209 (.node 3041 .empty .empty) .empty) (.node 3761 .empty .empty)) (.node 6329 (.node 5939 (.node 4337 .empty .empty) .empty) (.node 6353 .empty .empty))) (.node 8627 (.node 7589 (.node 7577 (.node 7211 .empty .empty) .empty) (.node 8039 .empty .empty)) (.node 10691 (.node 9839 .empty .empty) (.node 11027 .empty .empty)))) (.node 16319 (.node 15101 (.node 13697 (.node 13367 (.node 12959 .empty .empty) .empty) (.node 13883 .empty .empty)) (.node 15797 (.node 15377 .empty .empty) (.node 15923 .empty .empty))) (.node 19427 (.node 17327 (.node 16943 (.node 16481 .empty .empty) .empty) (.node 18749 .empty .empty)) (.node 22193 (.node 21143 .empty .empty) (.node 22943 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2699_ordered : SortedNatTree.ordered row2699 = true := by decide
set_option maxRecDepth 4096 in
theorem row2699_values : SortedNatTree.values row2699 = Data.row2699 := rfl
theorem row2699_checked : SortedNatTree.contains row2699 = Data.row2699.contains := by
  rw [SortedNatTree.contains_eq_list row2699 row2699_ordered, row2699_values]
def row2707 : SortedNatTree.Tree := (.node 11443 (.node 5857 (.node 4261 (.node 3463 (.node 2887 (.node 2791 .empty .empty) (.node 3253 .empty .empty)) (.node 3793 (.node 3517 .empty .empty) .empty)) (.node 5413 (.node 4507 (.node 4273 .empty .empty) (.node 4909 .empty .empty)) (.node 5701 (.node 5557 .empty .empty) .empty))) (.node 8053 (.node 6961 (.node 6229 (.node 6079 .empty .empty) (.node 6607 .empty .empty)) (.node 7789 (.node 7759 .empty .empty) .empty)) (.node 10333 (.node 9973 (.node 8599 .empty .empty) .empty) (.node 10729 (.node 10723 .empty .empty) .empty)))) (.node 18493 (.node 14503 (.node 13249 (.node 12157 (.node 12109 .empty .empty) (.node 12487 .empty .empty)) (.node 14281 (.node 14107 .empty .empty) .empty)) (.node 15583 (.node 15139 (.node 15073 .empty .empty) .empty) (.node 18313 (.node 16903 .empty .empty) .empty))) (.node 20743 (.node 19819 (.node 19489 (.node 19231 .empty .empty) (.node 19609 .empty .empty)) (.node 20641 (.node 20011 .empty .empty) .empty)) (.node 21817 (.node 21601 (.node 21019 .empty .empty) .empty) (.node 22717 (.node 22381 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2707_ordered : SortedNatTree.ordered row2707 = true := by decide
set_option maxRecDepth 4096 in
theorem row2707_values : SortedNatTree.values row2707 = Data.row2707 := rfl
theorem row2707_checked : SortedNatTree.contains row2707 = Data.row2707.contains := by
  rw [SortedNatTree.contains_eq_list row2707 row2707_ordered, row2707_values]
def row2711 : SortedNatTree.Tree := (.node 10667 (.node 7151 (.node 4421 (.node 3413 (.node 3137 (.node 2963 .empty .empty) .empty) (.node 4211 (.node 3851 .empty .empty) .empty)) (.node 5741 (.node 5651 (.node 5441 .empty .empty) .empty) (.node 6491 (.node 6113 .empty .empty) .empty))) (.node 9221 (.node 7817 (.node 7589 (.node 7457 .empty .empty) .empty) (.node 8747 (.node 7877 .empty .empty) .empty)) (.node 9431 (.node 9413 (.node 9371 .empty .empty) .empty) (.node 10061 (.node 9923 .empty .empty) .empty)))) (.node 17387 (.node 12269 (.node 11783 (.node 11657 (.node 11261 .empty .empty) .empty) (.node 12251 (.node 12203 .empty .empty) .empty)) (.node 16217 (.node 14411 (.node 12917 .empty .empty) .empty) (.node 16829 (.node 16661 .empty .empty) .empty))) (.node 21701 (.node 18521 (.node 18047 (.node 17417 .empty .empty) .empty) (.node 20357 (.node 18671 .empty .empty) .empty)) (.node 22961 (.node 22811 (.node 22283 .empty .empty) .empty) (.node 23189 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2711_ordered : SortedNatTree.ordered row2711 = true := by decide
set_option maxRecDepth 4096 in
theorem row2711_values : SortedNatTree.values row2711 = Data.row2711 := rfl
theorem row2711_checked : SortedNatTree.contains row2711 = Data.row2711.contains := by
  rw [SortedNatTree.contains_eq_list row2711 row2711_ordered, row2711_values]
def row2713 : SortedNatTree.Tree := (.node 10321 (.node 7213 (.node 5443 (.node 3517 (.node 3433 (.node 2797 .empty .empty) .empty) (.node 3889 (.node 3739 .empty .empty) .empty)) (.node 6247 (.node 5869 (.node 5749 .empty .empty) .empty) (.node 7207 (.node 6469 .empty .empty) .empty))) (.node 9397 (.node 8737 (.node 8389 (.node 8191 .empty .empty) .empty) (.node 9199 (.node 9187 .empty .empty) .empty)) (.node 9817 (.node 9613 (.node 9403 .empty .empty) .empty) (.node 10141 .empty .empty)))) (.node 15823 (.node 14431 (.node 12583 (.node 12553 (.node 10909 .empty .empty) .empty) (.node 13807 (.node 13477 .empty .empty) .empty)) (.node 15313 (.node 15013 (.node 14551 .empty .empty) .empty) (.node 15817 .empty .empty))) (.node 19603 (.node 17419 (.node 17257 (.node 16189 .empty .empty) .empty) (.node 18211 (.node 17539 .empty .empty) .empty)) (.node 22783 (.node 22039 (.node 20011 .empty .empty) .empty) (.node 23131 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2713_ordered : SortedNatTree.ordered row2713 = true := by decide
set_option maxRecDepth 4096 in
theorem row2713_values : SortedNatTree.values row2713 = Data.row2713 := rfl
theorem row2713_checked : SortedNatTree.contains row2713 = Data.row2713.contains := by
  rw [SortedNatTree.contains_eq_list row2713 row2713_ordered, row2713_values]
def row2719 : SortedNatTree.Tree := (.node 14479 (.node 9631 (.node 7129 (.node 4549 (.node 3391 .empty .empty) (.node 6121 .empty .empty)) (.node 8599 (.node 7669 .empty .empty) (.node 8713 .empty .empty))) (.node 12073 (.node 10567 (.node 9817 .empty .empty) (.node 10687 .empty .empty)) (.node 13729 (.node 12379 .empty .empty) (.node 14011 .empty .empty)))) (.node 17299 (.node 15349 (.node 15031 (.node 14563 .empty .empty) (.node 15199 .empty .empty)) (.node 16369 (.node 16273 .empty .empty) (.node 16987 .empty .empty))) (.node 19603 (.node 17749 (.node 17713 .empty .empty) (.node 18979 .empty .empty)) (.node 22741 (.node 21961 .empty .empty) (.node 23203 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2719_ordered : SortedNatTree.ordered row2719 = true := by decide
set_option maxRecDepth 4096 in
theorem row2719_values : SortedNatTree.values row2719 = Data.row2719 := rfl
theorem row2719_checked : SortedNatTree.contains row2719 = Data.row2719.contains := by
  rw [SortedNatTree.contains_eq_list row2719 row2719_ordered, row2719_values]
def row2729 : SortedNatTree.Tree := (.node 13691 (.node 9719 (.node 6131 (.node 4409 (.node 3191 (.node 3023 .empty .empty) .empty) (.node 5813 .empty .empty)) (.node 6551 (.node 6473 .empty .empty) (.node 8081 .empty .empty))) (.node 12659 (.node 11213 (.node 10973 (.node 10193 .empty .empty) .empty) (.node 11789 .empty .empty)) (.node 13457 (.node 13037 .empty .empty) (.node 13649 .empty .empty)))) (.node 18089 (.node 15767 (.node 15173 (.node 14897 (.node 13829 .empty .empty) .empty) (.node 15581 .empty .empty)) (.node 16223 (.node 15809 .empty .empty) (.node 17123 .empty .empty))) (.node 20357 (.node 19181 (.node 18341 .empty .empty) (.node 19979 .empty .empty)) (.node 22193 (.node 21599 .empty .empty) (.node 22961 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2729_ordered : SortedNatTree.ordered row2729 = true := by decide
set_option maxRecDepth 4096 in
theorem row2729_values : SortedNatTree.values row2729 = Data.row2729 := rfl
theorem row2729_checked : SortedNatTree.contains row2729 = Data.row2729.contains := by
  rw [SortedNatTree.contains_eq_list row2729 row2729_ordered, row2729_values]
def row2731 : SortedNatTree.Tree := (.node 11161 (.node 7867 (.node 4657 (.node 3613 (.node 3571 (.node 3301 .empty .empty) .empty) (.node 4423 (.node 4129 .empty .empty) .empty)) (.node 6451 (.node 5641 (.node 4759 .empty .empty) .empty) (.node 7753 .empty .empty))) (.node 10333 (.node 9157 (.node 8737 (.node 7993 .empty .empty) .empty) (.node 9661 (.node 9619 .empty .empty) .empty)) (.node 10639 (.node 10597 (.node 10531 .empty .empty) .empty) (.node 10723 .empty .empty)))) (.node 16447 (.node 13723 (.node 12391 (.node 11329 (.node 11287 .empty .empty) .empty) (.node 12841 (.node 12547 .empty .empty) .empty)) (.node 15121 (.node 15091 (.node 14221 .empty .empty) .empty) (.node 15439 .empty .empty))) (.node 18229 (.node 17569 (.node 17539 (.node 17137 .empty .empty) .empty) (.node 18049 (.node 17881 .empty .empty) .empty)) (.node 20773 (.node 18691 (.node 18553 .empty .empty) .empty) (.node 21157 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2731_ordered : SortedNatTree.ordered row2731 = true := by decide
set_option maxRecDepth 4096 in
theorem row2731_values : SortedNatTree.values row2731 = Data.row2731 := rfl
theorem row2731_checked : SortedNatTree.contains row2731 = Data.row2731.contains := by
  rw [SortedNatTree.contains_eq_list row2731 row2731_ordered, row2731_values]
def row2741 : SortedNatTree.Tree := (.node 10589 (.node 6353 (.node 3917 (.node 3119 (.node 2963 (.node 2837 .empty .empty) .empty) (.node 3389 (.node 3329 .empty .empty) .empty)) (.node 4649 (.node 4253 (.node 4133 .empty .empty) .empty) (.node 6113 (.node 4799 .empty .empty) .empty))) (.node 7457 (.node 6899 (.node 6737 (.node 6689 .empty .empty) .empty) (.node 7331 (.node 6911 .empty .empty) .empty)) (.node 8747 (.node 8537 (.node 7757 .empty .empty) .empty) (.node 9851 (.node 9467 .empty .empty) .empty)))) (.node 17183 (.node 13709 (.node 12071 (.node 11597 (.node 10799 .empty .empty) .empty) (.node 12899 (.node 12491 .empty .empty) .empty)) (.node 14549 (.node 14411 (.node 13799 .empty .empty) .empty) (.node 16433 (.node 15131 .empty .empty) .empty))) (.node 20921 (.node 20123 (.node 18371 (.node 18287 .empty .empty) .empty) (.node 20897 (.node 20681 .empty .empty) .empty)) (.node 22193 (.node 21563 (.node 21017 .empty .empty) .empty) (.node 22271 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2741_ordered : SortedNatTree.ordered row2741 = true := by decide
set_option maxRecDepth 4096 in
theorem row2741_values : SortedNatTree.values row2741 = Data.row2741 := rfl
theorem row2741_checked : SortedNatTree.contains row2741 = Data.row2741.contains := by
  rw [SortedNatTree.contains_eq_list row2741 row2741_ordered, row2741_values]
def row2749 : SortedNatTree.Tree := (.node 13537 (.node 8017 (.node 4297 (.node 4021 (.node 3643 (.node 3541 .empty .empty) .empty) (.node 4261 .empty .empty)) (.node 7219 (.node 6067 (.node 4993 .empty .empty) .empty) (.node 7741 .empty .empty))) (.node 10501 (.node 9781 (.node 9067 (.node 8923 .empty .empty) .empty) (.node 10477 .empty .empty)) (.node 12073 (.node 11467 .empty .empty) (.node 13183 .empty .empty)))) (.node 19231 (.node 15241 (.node 14449 (.node 14281 (.node 14149 .empty .empty) .empty) (.node 14737 .empty .empty)) (.node 18097 (.node 17449 (.node 17191 .empty .empty) .empty) (.node 18439 .empty .empty))) (.node 22567 (.node 21391 (.node 20599 (.node 19963 .empty .empty) .empty) (.node 22051 .empty .empty)) (.node 22993 (.node 22741 .empty .empty) (.node 23197 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2749_ordered : SortedNatTree.ordered row2749 = true := by decide
set_option maxRecDepth 4096 in
theorem row2749_values : SortedNatTree.values row2749 = Data.row2749 := rfl
theorem row2749_checked : SortedNatTree.contains row2749 = Data.row2749.contains := by
  rw [SortedNatTree.contains_eq_list row2749 row2749_ordered, row2749_values]
def row2753 : SortedNatTree.Tree := (.node 12473 (.node 7487 (.node 5171 (.node 3203 (.node 2963 (.node 2927 .empty .empty) (.node 3089 .empty .empty)) (.node 4583 (.node 3371 .empty .empty) (.node 4799 .empty .empty))) (.node 7001 (.node 6287 (.node 6101 .empty .empty) (.node 6791 .empty .empty)) (.node 7481 (.node 7457 .empty .empty) .empty))) (.node 9293 (.node 8831 (.node 7919 (.node 7529 .empty .empty) (.node 8597 .empty .empty)) (.node 9059 (.node 8933 .empty .empty) (.node 9203 .empty .empty))) (.node 12269 (.node 9923 (.node 9689 .empty .empty) (.node 11579 .empty .empty)) (.node 12347 (.node 12329 .empty .empty) .empty)))) (.node 16253 (.node 14771 (.node 14243 (.node 13721 (.node 12497 .empty .empty) (.node 14051 .empty .empty)) (.node 14387 (.node 14321 .empty .empty) (.node 14519 .empty .empty))) (.node 15569 (.node 15101 (.node 14867 .empty .empty) (.node 15413 .empty .empty)) (.node 15791 (.node 15737 .empty .empty) .empty))) (.node 19739 (.node 18041 (.node 17477 (.node 16661 .empty .empty) (.node 17891 .empty .empty)) (.node 19571 (.node 18089 .empty .empty) .empty)) (.node 22247 (.node 20939 (.node 20249 .empty .empty) (.node 21611 .empty .empty)) (.node 22727 (.node 22643 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2753_ordered : SortedNatTree.ordered row2753 = true := by decide
set_option maxRecDepth 4096 in
theorem row2753_values : SortedNatTree.values row2753 = Data.row2753 := rfl
theorem row2753_checked : SortedNatTree.contains row2753 = Data.row2753.contains := by
  rw [SortedNatTree.contains_eq_list row2753 row2753_ordered, row2753_values]
def row2767 : SortedNatTree.Tree := (.node 9661 (.node 5737 (.node 4513 (.node 3889 (.node 3187 (.node 3109 .empty .empty) (.node 3733 .empty .empty)) (.node 4099 (.node 3967 .empty .empty) .empty)) (.node 5107 (.node 5059 (.node 4783 .empty .empty) .empty) (.node 5689 (.node 5119 .empty .empty) .empty))) (.node 7561 (.node 7459 (.node 6679 (.node 6361 .empty .empty) (.node 6967 .empty .empty)) (.node 7537 (.node 7477 .empty .empty) .empty)) (.node 7933 (.node 7723 (.node 7621 .empty .empty) .empty) (.node 8527 (.node 8419 .empty .empty) .empty)))) (.node 15679 (.node 13687 (.node 11971 (.node 10627 (.node 9721 .empty .empty) (.node 10789 .empty .empty)) (.node 13399 (.node 12781 .empty .empty) .empty)) (.node 14923 (.node 14551 (.node 14293 .empty .empty) .empty) (.node 15619 (.node 15121 .empty .empty) .empty))) (.node 19543 (.node 18289 (.node 16477 (.node 16249 .empty .empty) .empty) (.node 18919 (.node 18433 .empty .empty) .empty)) (.node 21397 (.node 20551 (.node 20071 .empty .empty) .empty) (.node 23203 (.node 22189 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2767_ordered : SortedNatTree.ordered row2767 = true := by decide
set_option maxRecDepth 4096 in
theorem row2767_values : SortedNatTree.values row2767 = Data.row2767 := rfl
theorem row2767_checked : SortedNatTree.contains row2767 = Data.row2767.contains := by
  rw [SortedNatTree.contains_eq_list row2767 row2767_ordered, row2767_values]
def row2777 : SortedNatTree.Tree := (.node 14243 (.node 9533 (.node 6473 (.node 4973 (.node 4793 (.node 4289 .empty .empty) .empty) (.node 6173 (.node 5477 .empty .empty) .empty)) (.node 8087 (.node 8039 (.node 6971 .empty .empty) .empty) (.node 8819 (.node 8291 .empty .empty) .empty))) (.node 11597 (.node 10331 (.node 10163 (.node 9551 .empty .empty) .empty) (.node 10799 (.node 10781 .empty .empty) .empty)) (.node 13691 (.node 13127 (.node 12377 .empty .empty) .empty) (.node 13859 (.node 13781 .empty .empty) .empty)))) (.node 17681 (.node 15497 (.node 14561 (.node 14447 (.node 14387 .empty .empty) .empty) (.node 15401 (.node 15161 .empty .empty) .empty)) (.node 17183 (.node 16619 (.node 15797 .empty .empty) .empty) (.node 17519 (.node 17333 .empty .empty) .empty))) (.node 20249 (.node 18719 (.node 18671 (.node 17807 .empty .empty) .empty) (.node 20123 (.node 19073 .empty .empty) .empty)) (.node 21383 (.node 20903 (.node 20627 .empty .empty) .empty) (.node 22811 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2777_ordered : SortedNatTree.ordered row2777 = true := by decide
set_option maxRecDepth 4096 in
theorem row2777_values : SortedNatTree.values row2777 = Data.row2777 := rfl
theorem row2777_checked : SortedNatTree.contains row2777 = Data.row2777.contains := by
  rw [SortedNatTree.contains_eq_list row2777 row2777_ordered, row2777_values]
def row2789 : SortedNatTree.Tree := (.node 8831 (.node 5471 (.node 4133 (.node 3461 (.node 3323 (.node 2861 .empty .empty) .empty) (.node 4007 .empty .empty)) (.node 4703 (.node 4673 .empty .empty) (.node 5051 .empty .empty))) (.node 7691 (.node 6563 (.node 6299 (.node 5639 .empty .empty) .empty) (.node 6689 .empty .empty)) (.node 8447 (.node 8369 .empty .empty) (.node 8537 .empty .empty)))) (.node 15887 (.node 12197 (.node 9497 (.node 9173 (.node 8951 .empty .empty) .empty) (.node 10601 .empty .empty)) (.node 14543 (.node 13841 .empty .empty) (.node 15671 .empty .empty))) (.node 21017 (.node 17477 (.node 16691 (.node 16061 .empty .empty) .empty) (.node 18443 .empty .empty)) (.node 21107 (.node 21089 .empty .empty) (.node 22901 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2789_ordered : SortedNatTree.ordered row2789 = true := by decide
set_option maxRecDepth 4096 in
theorem row2789_values : SortedNatTree.values row2789 = Data.row2789 := rfl
theorem row2789_checked : SortedNatTree.contains row2789 = Data.row2789.contains := by
  rw [SortedNatTree.contains_eq_list row2789 row2789_ordered, row2789_values]
def row2791 : SortedNatTree.Tree := (.node 10177 (.node 6883 (.node 5113 (.node 3967 (.node 3499 (.node 3001 .empty .empty) .empty) (.node 4297 .empty .empty)) (.node 6379 (.node 5281 (.node 5179 .empty .empty) .empty) (.node 6607 .empty .empty))) (.node 8287 (.node 7489 (.node 7369 (.node 6961 .empty .empty) .empty) (.node 8179 .empty .empty)) (.node 9511 (.node 8647 .empty .empty) (.node 9649 .empty .empty)))) (.node 16843 (.node 12073 (.node 11353 (.node 10789 (.node 10333 .empty .empty) .empty) (.node 11467 .empty .empty)) (.node 15619 (.node 15541 (.node 12391 .empty .empty) .empty) (.node 16759 .empty .empty))) (.node 19207 (.node 18313 (.node 18097 (.node 17863 .empty .empty) .empty) (.node 18523 .empty .empty)) (.node 21943 (.node 20047 .empty .empty) (.node 21961 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2791_ordered : SortedNatTree.ordered row2791 = true := by decide
set_option maxRecDepth 4096 in
theorem row2791_values : SortedNatTree.values row2791 = Data.row2791 := rfl
theorem row2791_checked : SortedNatTree.contains row2791 = Data.row2791.contains := by
  rw [SortedNatTree.contains_eq_list row2791 row2791_ordered, row2791_values]
def row2797 : SortedNatTree.Tree := (.node 12583 (.node 7207 (.node 5449 (.node 4663 (.node 4567 (.node 4561 .empty .empty) .empty) (.node 5023 (.node 4861 .empty .empty) .empty)) (.node 6367 (.node 6163 (.node 5953 .empty .empty) .empty) (.node 6841 .empty .empty))) (.node 9643 (.node 8017 (.node 7759 (.node 7681 .empty .empty) .empty) (.node 9181 (.node 8563 .empty .empty) .empty)) (.node 10111 (.node 10099 (.node 9871 .empty .empty) .empty) (.node 10867 .empty .empty)))) (.node 19009 (.node 17419 (.node 13933 (.node 13249 (.node 12967 .empty .empty) .empty) (.node 15493 (.node 13963 .empty .empty) .empty)) (.node 18973 (.node 17959 (.node 17497 .empty .empty) .empty) (.node 18979 .empty .empty))) (.node 22171 (.node 21397 (.node 21157 (.node 19051 .empty .empty) .empty) (.node 21613 .empty .empty)) (.node 23167 (.node 22447 (.node 22279 .empty .empty) .empty) (.node 23209 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2797_ordered : SortedNatTree.ordered row2797 = true := by decide
set_option maxRecDepth 4096 in
theorem row2797_values : SortedNatTree.values row2797 = Data.row2797 := rfl
theorem row2797_checked : SortedNatTree.contains row2797 = Data.row2797.contains := by
  rw [SortedNatTree.contains_eq_list row2797 row2797_ordered, row2797_values]
def row2801 : SortedNatTree.Tree := (.node 9743 (.node 5987 (.node 4073 (.node 3803 (.node 3221 (.node 2843 .empty .empty) (.node 3527 .empty .empty)) (.node 3917 (.node 3851 .empty .empty) .empty)) (.node 5417 (.node 4733 (.node 4229 .empty .empty) (.node 5051 .empty .empty)) (.node 5693 (.node 5573 .empty .empty) .empty))) (.node 7331 (.node 6803 (.node 6311 (.node 6101 .empty .empty) (.node 6689 .empty .empty)) (.node 7193 (.node 7043 .empty .empty) .empty)) (.node 8387 (.node 8087 (.node 7919 .empty .empty) (.node 8171 .empty .empty)) (.node 9491 (.node 9461 .empty .empty) .empty)))) (.node 16433 (.node 13187 (.node 11483 (.node 10667 (.node 9857 .empty .empty) (.node 11369 .empty .empty)) (.node 12011 (.node 11987 .empty .empty) .empty)) (.node 14759 (.node 13829 (.node 13451 .empty .empty) (.node 14633 .empty .empty)) (.node 16349 (.node 15497 .empty .empty) .empty))) (.node 20177 (.node 18719 (.node 17471 (.node 16619 .empty .empty) (.node 17837 .empty .empty)) (.node 19571 (.node 19403 .empty .empty) .empty)) (.node 22613 (.node 21713 (.node 21599 .empty .empty) .empty) (.node 23201 (.node 22901 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2801_ordered : SortedNatTree.ordered row2801 = true := by decide
set_option maxRecDepth 4096 in
theorem row2801_values : SortedNatTree.values row2801 = Data.row2801 := rfl
theorem row2801_checked : SortedNatTree.contains row2801 = Data.row2801.contains := by
  rw [SortedNatTree.contains_eq_list row2801 row2801_ordered, row2801_values]
def row2803 : SortedNatTree.Tree := (.node 9091 (.node 5851 (.node 4507 (.node 3769 (.node 3631 (.node 3079 .empty .empty) .empty) (.node 4423 (.node 4051 .empty .empty) .empty)) (.node 4969 (.node 4813 (.node 4549 .empty .empty) .empty) (.node 5701 (.node 5107 .empty .empty) .empty))) (.node 7951 (.node 6733 (.node 6361 (.node 6199 .empty .empty) .empty) (.node 7717 (.node 7549 .empty .empty) .empty)) (.node 8971 (.node 8377 (.node 7993 .empty .empty) .empty) (.node 9049 .empty .empty)))) (.node 15859 (.node 13291 (.node 10567 (.node 9883 (.node 9127 .empty .empty) .empty) (.node 12763 (.node 11443 .empty .empty) .empty)) (.node 14341 (.node 14143 (.node 13381 .empty .empty) .empty) (.node 15727 (.node 15031 .empty .empty) .empty))) (.node 17929 (.node 16903 (.node 16657 (.node 16573 .empty .empty) .empty) (.node 17839 (.node 17449 .empty .empty) .empty)) (.node 19387 (.node 19381 (.node 18217 .empty .empty) .empty) (.node 20809 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2803_ordered : SortedNatTree.ordered row2803 = true := by decide
set_option maxRecDepth 4096 in
theorem row2803_values : SortedNatTree.values row2803 = Data.row2803 := rfl
theorem row2803_checked : SortedNatTree.contains row2803 = Data.row2803.contains := by
  rw [SortedNatTree.contains_eq_list row2803 row2803_ordered, row2803_values]
def row2819 : SortedNatTree.Tree := (.node 13487 (.node 5927 (.node 5039 (.node 4079 (.node 3617 (.node 2897 .empty .empty) .empty) (.node 4787 .empty .empty)) (.node 5477 (.node 5381 (.node 5351 .empty .empty) .empty) (.node 5879 .empty .empty))) (.node 8081 (.node 7523 (.node 7187 (.node 6653 .empty .empty) .empty) (.node 7607 .empty .empty)) (.node 11423 (.node 9767 (.node 9203 .empty .empty) .empty) (.node 12539 .empty .empty)))) (.node 18047 (.node 16301 (.node 14411 (.node 14177 (.node 13799 .empty .empty) .empty) (.node 16001 .empty .empty)) (.node 17981 (.node 17921 (.node 16829 .empty .empty) .empty) (.node 18041 .empty .empty))) (.node 22031 (.node 21407 (.node 20183 (.node 19139 .empty .empty) .empty) (.node 21587 .empty .empty)) (.node 22787 (.node 22307 (.node 22073 .empty .empty) .empty) (.node 23039 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2819_ordered : SortedNatTree.ordered row2819 = true := by decide
set_option maxRecDepth 4096 in
theorem row2819_values : SortedNatTree.values row2819 = Data.row2819 := rfl
theorem row2819_checked : SortedNatTree.contains row2819 = Data.row2819.contains := by
  rw [SortedNatTree.contains_eq_list row2819 row2819_ordered, row2819_values]
def row2833 : SortedNatTree.Tree := (.node 12511 (.node 6361 (.node 4759 (.node 4021 (.node 3229 (.node 3037 .empty .empty) .empty) (.node 4591 .empty .empty)) (.node 5851 (.node 5683 (.node 5113 .empty .empty) .empty) (.node 5869 .empty .empty))) (.node 9613 (.node 8389 (.node 7723 (.node 7297 .empty .empty) .empty) (.node 8581 .empty .empty)) (.node 10177 (.node 10099 (.node 9631 .empty .empty) .empty) (.node 11719 .empty .empty)))) (.node 17389 (.node 15601 (.node 13873 (.node 13291 (.node 12517 .empty .empty) .empty) (.node 14071 .empty .empty)) (.node 16567 (.node 16249 (.node 15787 .empty .empty) .empty) (.node 17137 .empty .empty))) (.node 20161 (.node 18181 (.node 18097 (.node 17449 .empty .empty) .empty) (.node 19867 .empty .empty)) (.node 22549 (.node 20959 (.node 20407 .empty .empty) .empty) (.node 22651 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2833_ordered : SortedNatTree.ordered row2833 = true := by decide
set_option maxRecDepth 4096 in
theorem row2833_values : SortedNatTree.values row2833 = Data.row2833 := rfl
theorem row2833_checked : SortedNatTree.contains row2833 = Data.row2833.contains := by
  rw [SortedNatTree.contains_eq_list row2833 row2833_ordered, row2833_values]
def row2837 : SortedNatTree.Tree := (.node 10859 (.node 7691 (.node 4817 (.node 3407 (.node 3299 (.node 3089 .empty .empty) .empty) (.node 3767 (.node 3617 .empty .empty) .empty)) (.node 5297 (.node 5153 (.node 5003 .empty .empty) .empty) (.node 7001 (.node 6449 .empty .empty) .empty))) (.node 9419 (.node 9161 (.node 7793 (.node 7757 .empty .empty) .empty) (.node 9311 (.node 9209 .empty .empty) .empty)) (.node 10463 (.node 10289 (.node 9431 .empty .empty) .empty) (.node 10733 .empty .empty)))) (.node 15569 (.node 13721 (.node 11579 (.node 11423 (.node 11027 .empty .empty) .empty) (.node 12329 (.node 12203 .empty .empty) .empty)) (.node 14891 (.node 14747 (.node 13841 .empty .empty) .empty) (.node 15461 .empty .empty))) (.node 20231 (.node 15791 (.node 15749 (.node 15647 .empty .empty) .empty) (.node 20117 (.node 17033 .empty .empty) .empty)) (.node 21929 (.node 21149 (.node 20771 .empty .empty) .empty) (.node 22787 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2837_ordered : SortedNatTree.ordered row2837 = true := by decide
set_option maxRecDepth 4096 in
theorem row2837_values : SortedNatTree.values row2837 = Data.row2837 := rfl
theorem row2837_checked : SortedNatTree.contains row2837 = Data.row2837.contains := by
  rw [SortedNatTree.contains_eq_list row2837 row2837_ordered, row2837_values]
def row2843 : SortedNatTree.Tree := (.node 10667 (.node 6047 (.node 4253 (.node 3389 (.node 3203 (.node 3089 .empty .empty) .empty) (.node 3989 (.node 3947 .empty .empty) .empty)) (.node 5783 (.node 5657 (.node 4817 .empty .empty) .empty) (.node 5807 .empty .empty))) (.node 9473 (.node 8171 (.node 7841 (.node 6899 .empty .empty) .empty) (.node 8831 (.node 8273 .empty .empty) .empty)) (.node 10427 (.node 10139 (.node 9857 .empty .empty) .empty) (.node 10457 .empty .empty)))) (.node 17669 (.node 14321 (.node 12653 (.node 12569 (.node 11939 .empty .empty) .empty) (.node 13313 (.node 13163 .empty .empty) .empty)) (.node 16943 (.node 15149 (.node 15137 .empty .empty) .empty) (.node 17471 .empty .empty))) (.node 21407 (.node 19463 (.node 19163 (.node 17909 .empty .empty) .empty) (.node 21227 (.node 20717 .empty .empty) .empty)) (.node 22691 (.node 22397 (.node 22277 .empty .empty) .empty) (.node 23099 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2843_ordered : SortedNatTree.ordered row2843 = true := by decide
set_option maxRecDepth 4096 in
theorem row2843_values : SortedNatTree.values row2843 = Data.row2843 := rfl
theorem row2843_checked : SortedNatTree.contains row2843 = Data.row2843.contains := by
  rw [SortedNatTree.contains_eq_list row2843 row2843_ordered, row2843_values]
def row2851 : SortedNatTree.Tree := (.node 11839 (.node 7219 (.node 5119 (.node 3733 (.node 3313 (.node 3187 .empty .empty) .empty) (.node 5107 (.node 4951 .empty .empty) .empty)) (.node 6277 (.node 6091 (.node 5737 .empty .empty) .empty) (.node 6949 .empty .empty))) (.node 10831 (.node 9697 (.node 9613 (.node 8647 .empty .empty) .empty) (.node 10729 (.node 9739 .empty .empty) .empty)) (.node 11527 (.node 11257 (.node 10939 .empty .empty) .empty) (.node 11719 .empty .empty)))) (.node 16141 (.node 14887 (.node 12739 (.node 12511 (.node 12007 .empty .empty) .empty) (.node 14449 (.node 13879 .empty .empty) .empty)) (.node 15601 (.node 15427 (.node 15139 .empty .empty) .empty) (.node 15913 .empty .empty))) (.node 20047 (.node 18913 (.node 16921 (.node 16879 .empty .empty) .empty) (.node 19609 (.node 19423 .empty .empty) .empty)) (.node 22483 (.node 21589 (.node 21559 .empty .empty) .empty) (.node 22741 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2851_ordered : SortedNatTree.ordered row2851 = true := by decide
set_option maxRecDepth 4096 in
theorem row2851_values : SortedNatTree.values row2851 = Data.row2851 := rfl
theorem row2851_checked : SortedNatTree.contains row2851 = Data.row2851.contains := by
  rw [SortedNatTree.contains_eq_list row2851 row2851_ordered, row2851_values]
def row2857 : SortedNatTree.Tree := (.node 11677 (.node 4957 (.node 3709 (.node 3229 (.node 3163 (.node 3049 .empty .empty) .empty) (.node 3499 (.node 3469 .empty .empty) .empty)) (.node 4549 (.node 4519 (.node 3907 .empty .empty) .empty) (.node 4783 (.node 4561 .empty .empty) .empty))) (.node 7933 (.node 6091 (.node 5881 (.node 5641 .empty .empty) .empty) (.node 7699 (.node 7573 .empty .empty) .empty)) (.node 10111 (.node 9043 (.node 8941 .empty .empty) .empty) (.node 10861 .empty .empty)))) (.node 15919 (.node 14071 (.node 12841 (.node 12781 (.node 12109 .empty .empty) .empty) (.node 13759 (.node 13693 .empty .empty) .empty)) (.node 15427 (.node 14983 (.node 14389 .empty .empty) .empty) (.node 15907 .empty .empty))) (.node 20857 (.node 18787 (.node 16447 (.node 16369 .empty .empty) .empty) (.node 19543 (.node 19009 .empty .empty) .empty)) (.node 22051 (.node 21991 (.node 21727 .empty .empty) .empty) (.node 22369 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2857_ordered : SortedNatTree.ordered row2857 = true := by decide
set_option maxRecDepth 4096 in
theorem row2857_values : SortedNatTree.values row2857 = Data.row2857 := rfl
theorem row2857_checked : SortedNatTree.contains row2857 = Data.row2857.contains := by
  rw [SortedNatTree.contains_eq_list row2857 row2857_ordered, row2857_values]
def row2861 : SortedNatTree.Tree := (.node 11783 (.node 7499 (.node 5693 (.node 3911 (.node 3671 (.node 3257 .empty .empty) .empty) (.node 4919 .empty .empty)) (.node 6719 (.node 5861 .empty .empty) (.node 7013 .empty .empty))) (.node 9743 (.node 9137 (.node 8171 (.node 7649 .empty .empty) .empty) (.node 9281 .empty .empty)) (.node 11213 (.node 11159 .empty .empty) (.node 11321 .empty .empty)))) (.node 18719 (.node 16451 (.node 12161 (.node 12119 (.node 11831 .empty .empty) .empty) (.node 12689 .empty .empty)) (.node 17729 (.node 17519 .empty .empty) (.node 18047 .empty .empty))) (.node 22619 (.node 20543 (.node 20357 (.node 20147 .empty .empty) .empty) (.node 20903 .empty .empty)) (.node 22751 (.node 22679 .empty .empty) (.node 23081 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2861_ordered : SortedNatTree.ordered row2861 = true := by decide
set_option maxRecDepth 4096 in
theorem row2861_values : SortedNatTree.values row2861 = Data.row2861 := rfl
theorem row2861_checked : SortedNatTree.contains row2861 = Data.row2861.contains := by
  rw [SortedNatTree.contains_eq_list row2861 row2861_ordered, row2861_values]
def row2879 : SortedNatTree.Tree := (.node 10193 (.node 5867 (.node 5303 (.node 4397 (.node 3581 (.node 3449 .empty .empty) .empty) (.node 5171 (.node 4673 .empty .empty) .empty)) (.node 5573 (.node 5531 (.node 5387 .empty .empty) .empty) (.node 5693 .empty .empty))) (.node 6959 (.node 6701 (.node 6221 (.node 6143 .empty .empty) .empty) (.node 6899 .empty .empty)) (.node 8741 (.node 7883 (.node 7151 .empty .empty) .empty) (.node 9941 .empty .empty)))) (.node 16787 (.node 12809 (.node 10631 (.node 10313 (.node 10253 .empty .empty) .empty) (.node 12161 .empty .empty)) (.node 15773 (.node 14447 (.node 13763 .empty .empty) .empty) (.node 16529 .empty .empty))) (.node 21419 (.node 20231 (.node 20021 (.node 19403 .empty .empty) .empty) (.node 21089 .empty .empty)) (.node 23057 (.node 22511 (.node 22307 .empty .empty) .empty) (.node 23087 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2879_ordered : SortedNatTree.ordered row2879 = true := by decide
set_option maxRecDepth 4096 in
theorem row2879_values : SortedNatTree.values row2879 = Data.row2879 := rfl
theorem row2879_checked : SortedNatTree.contains row2879 = Data.row2879.contains := by
  rw [SortedNatTree.contains_eq_list row2879 row2879_ordered, row2879_values]
def row2887 : SortedNatTree.Tree := (.node 14593 (.node 9439 (.node 5653 (.node 4273 (.node 4231 (.node 2917 .empty .empty) .empty) (.node 4987 (.node 4597 .empty .empty) .empty)) (.node 7573 (.node 6733 (.node 6709 .empty .empty) .empty) (.node 9001 .empty .empty))) (.node 12973 (.node 12763 (.node 12511 (.node 9511 .empty .empty) .empty) (.node 12967 (.node 12829 .empty .empty) .empty)) (.node 14011 (.node 13219 (.node 13171 .empty .empty) .empty) (.node 14431 .empty .empty)))) (.node 19477 (.node 17041 (.node 15277 (.node 14851 (.node 14821 .empty .empty) .empty) (.node 16651 (.node 16369 .empty .empty) .empty)) (.node 17929 (.node 17713 (.node 17491 .empty .empty) .empty) (.node 18181 .empty .empty))) (.node 21859 (.node 20101 (.node 20071 (.node 19813 .empty .empty) .empty) (.node 20359 .empty .empty)) (.node 22573 (.node 22129 (.node 21943 .empty .empty) .empty) (.node 22669 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2887_ordered : SortedNatTree.ordered row2887 = true := by decide
set_option maxRecDepth 4096 in
theorem row2887_values : SortedNatTree.values row2887 = Data.row2887 := rfl
theorem row2887_checked : SortedNatTree.contains row2887 = Data.row2887.contains := by
  rw [SortedNatTree.contains_eq_list row2887 row2887_ordered, row2887_values]
def row2897 : SortedNatTree.Tree := (.node 9551 (.node 6329 (.node 4493 (.node 3947 (.node 3209 (.node 3023 .empty .empty) .empty) (.node 4289 .empty .empty)) (.node 5501 (.node 4877 (.node 4721 .empty .empty) .empty) (.node 5867 .empty .empty))) (.node 8867 (.node 7307 (.node 7211 (.node 6473 .empty .empty) .empty) (.node 7547 .empty .empty)) (.node 9257 (.node 9227 .empty .empty) (.node 9539 .empty .empty)))) (.node 18059 (.node 11057 (.node 10163 (.node 10151 (.node 10079 .empty .empty) .empty) (.node 10937 .empty .empty)) (.node 15053 (.node 11789 .empty .empty) (.node 15299 .empty .empty))) (.node 21521 (.node 19403 (.node 18911 (.node 18353 .empty .empty) .empty) (.node 20789 .empty .empty)) (.node 22343 (.node 22193 .empty .empty) (.node 22739 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2897_ordered : SortedNatTree.ordered row2897 = true := by decide
set_option maxRecDepth 4096 in
theorem row2897_values : SortedNatTree.values row2897 = Data.row2897 := rfl
theorem row2897_checked : SortedNatTree.contains row2897 = Data.row2897.contains := by
  rw [SortedNatTree.contains_eq_list row2897 row2897_ordered, row2897_values]
def row2903 : SortedNatTree.Tree := (.node 10667 (.node 7103 (.node 4721 (.node 4409 (.node 4373 (.node 2963 .empty .empty) .empty) (.node 4637 .empty .empty)) (.node 6701 (.node 6131 (.node 5927 .empty .empty) .empty) (.node 6779 .empty .empty))) (.node 8963 (.node 7457 (.node 7433 (.node 7121 .empty .empty) .empty) (.node 8123 .empty .empty)) (.node 9851 (.node 9539 (.node 9173 .empty .empty) .empty) (.node 10607 .empty .empty)))) (.node 17471 (.node 13799 (.node 13007 (.node 11909 (.node 11003 .empty .empty) .empty) (.node 13469 .empty .empty)) (.node 15101 (.node 14051 (.node 13841 .empty .empty) .empty) (.node 16001 .empty .empty))) (.node 19937 (.node 18773 (.node 18341 (.node 17747 .empty .empty) .empty) (.node 19853 .empty .empty)) (.node 22811 (.node 21839 (.node 20201 .empty .empty) .empty) (.node 22937 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2903_ordered : SortedNatTree.ordered row2903 = true := by decide
set_option maxRecDepth 4096 in
theorem row2903_values : SortedNatTree.values row2903 = Data.row2903 := rfl
theorem row2903_checked : SortedNatTree.contains row2903 = Data.row2903.contains := by
  rw [SortedNatTree.contains_eq_list row2903 row2903_ordered, row2903_values]
def row2909 : SortedNatTree.Tree := (.node 9419 (.node 6737 (.node 5153 (.node 4217 (.node 3557 (.node 3089 .empty .empty) .empty) (.node 4253 .empty .empty)) (.node 6269 (.node 5813 .empty .empty) (.node 6323 .empty .empty))) (.node 9011 (.node 7451 (.node 6983 .empty .empty) (.node 8669 .empty .empty)) (.node 9293 (.node 9227 .empty .empty) (.node 9311 .empty .empty)))) (.node 16103 (.node 10889 (.node 10091 (.node 9857 (.node 9689 .empty .empty) .empty) (.node 10853 .empty .empty)) (.node 15797 (.node 12899 .empty .empty) (.node 15881 .empty .empty))) (.node 22013 (.node 18917 (.node 16193 .empty .empty) (.node 21767 .empty .empty)) (.node 22937 (.node 22859 .empty .empty) (.node 22943 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2909_ordered : SortedNatTree.ordered row2909 = true := by decide
set_option maxRecDepth 4096 in
theorem row2909_values : SortedNatTree.values row2909 = Data.row2909 := rfl
theorem row2909_checked : SortedNatTree.contains row2909 = Data.row2909.contains := by
  rw [SortedNatTree.contains_eq_list row2909 row2909_ordered, row2909_values]
def row2917 : SortedNatTree.Tree := (.node 12829 (.node 7549 (.node 4909 (.node 3643 (.node 3631 (.node 3169 .empty .empty) .empty) (.node 4003 (.node 3733 .empty .empty) .empty)) (.node 5437 (.node 5227 (.node 5011 .empty .empty) .empty) (.node 5851 .empty .empty))) (.node 9859 (.node 8731 (.node 8581 (.node 8179 .empty .empty) .empty) (.node 9721 .empty .empty)) (.node 11149 (.node 10267 (.node 10099 .empty .empty) .empty) (.node 11443 .empty .empty)))) (.node 20749 (.node 18121 (.node 15661 (.node 14653 (.node 13567 .empty .empty) .empty) (.node 16759 .empty .empty)) (.node 19801 (.node 19249 (.node 19081 .empty .empty) .empty) (.node 20641 .empty .empty))) (.node 21523 (.node 21499 (.node 21163 (.node 20959 .empty .empty) .empty) (.node 21517 .empty .empty)) (.node 22111 (.node 22063 (.node 21841 .empty .empty) .empty) (.node 22129 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2917_ordered : SortedNatTree.ordered row2917 = true := by decide
set_option maxRecDepth 4096 in
theorem row2917_values : SortedNatTree.values row2917 = Data.row2917 := rfl
theorem row2917_checked : SortedNatTree.contains row2917 = Data.row2917.contains := by
  rw [SortedNatTree.contains_eq_list row2917 row2917_ordered, row2917_values]
def row2927 : SortedNatTree.Tree := (.node 10853 (.node 6257 (.node 5009 (.node 3593 (.node 3203 (.node 3137 .empty .empty) .empty) (.node 3863 .empty .empty)) (.node 5693 (.node 5531 .empty .empty) (.node 6029 .empty .empty))) (.node 8999 (.node 8009 (.node 6983 .empty .empty) (.node 8243 .empty .empty)) (.node 9227 (.node 9011 .empty .empty) (.node 10037 .empty .empty)))) (.node 18131 (.node 12611 (.node 11519 (.node 11057 .empty .empty) (.node 11987 .empty .empty)) (.node 13457 (.node 13103 .empty .empty) (.node 15263 .empty .empty))) (.node 22343 (.node 20849 (.node 18749 .empty .empty) (.node 21557 .empty .empty)) (.node 22751 (.node 22709 .empty .empty) (.node 23027 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2927_ordered : SortedNatTree.ordered row2927 = true := by decide
set_option maxRecDepth 4096 in
theorem row2927_values : SortedNatTree.values row2927 = Data.row2927 := rfl
theorem row2927_checked : SortedNatTree.contains row2927 = Data.row2927.contains := by
  rw [SortedNatTree.contains_eq_list row2927 row2927_ordered, row2927_values]
def row2939 : SortedNatTree.Tree := (.node 9677 (.node 5297 (.node 3671 (.node 3491 (.node 3167 (.node 2999 .empty .empty) .empty) (.node 3557 .empty .empty)) (.node 4217 (.node 4133 (.node 4019 .empty .empty) .empty) (.node 4679 .empty .empty))) (.node 7211 (.node 5981 (.node 5573 (.node 5417 .empty .empty) .empty) (.node 6863 .empty .empty)) (.node 9203 (.node 8783 (.node 8573 .empty .empty) .empty) (.node 9551 .empty .empty)))) (.node 14321 (.node 11789 (.node 11003 (.node 10847 (.node 9791 .empty .empty) .empty) (.node 11621 .empty .empty)) (.node 13709 (.node 12821 (.node 12041 .empty .empty) .empty) (.node 13877 .empty .empty))) (.node 19793 (.node 16661 (.node 16253 (.node 15473 .empty .empty) .empty) (.node 18701 .empty .empty)) (.node 21587 (.node 21269 (.node 20441 .empty .empty) .empty) (.node 22133 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2939_ordered : SortedNatTree.ordered row2939 = true := by decide
set_option maxRecDepth 4096 in
theorem row2939_values : SortedNatTree.values row2939 = Data.row2939 := rfl
theorem row2939_checked : SortedNatTree.contains row2939 = Data.row2939.contains := by
  rw [SortedNatTree.contains_eq_list row2939 row2939_ordered, row2939_values]
def row2953 : SortedNatTree.Tree := (.node 8461 (.node 6709 (.node 4357 (.node 3643 (.node 3433 (.node 3181 .empty .empty) .empty) (.node 4273 .empty .empty)) (.node 6271 (.node 6217 (.node 4957 .empty .empty) .empty) (.node 6679 .empty .empty))) (.node 7369 (.node 7213 (.node 6907 (.node 6733 .empty .empty) .empty) (.node 7309 .empty .empty)) (.node 7759 (.node 7591 (.node 7537 .empty .empty) .empty) (.node 7867 .empty .empty)))) (.node 15679 (.node 12517 (.node 10069 (.node 9649 (.node 9343 .empty .empty) .empty) (.node 10459 .empty .empty)) (.node 14629 (.node 14281 (.node 13633 .empty .empty) .empty) (.node 14683 .empty .empty))) (.node 18553 (.node 18217 (.node 16057 (.node 15919 .empty .empty) .empty) (.node 18313 .empty .empty)) (.node 20359 (.node 20101 (.node 19867 .empty .empty) .empty) (.node 21001 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2953_ordered : SortedNatTree.ordered row2953 = true := by decide
set_option maxRecDepth 4096 in
theorem row2953_values : SortedNatTree.values row2953 = Data.row2953 := rfl
theorem row2953_checked : SortedNatTree.contains row2953 = Data.row2953.contains := by
  rw [SortedNatTree.contains_eq_list row2953 row2953_ordered, row2953_values]
def row2957 : SortedNatTree.Tree := (.node 12611 (.node 8291 (.node 6101 (.node 4451 (.node 3449 (.node 3041 .empty .empty) .empty) (.node 4973 .empty .empty)) (.node 7307 (.node 6803 (.node 6221 .empty .empty) .empty) (.node 7451 .empty .empty))) (.node 9377 (.node 9281 (.node 8831 (.node 8543 .empty .empty) .empty) (.node 9323 .empty .empty)) (.node 9791 (.node 9551 .empty .empty) (.node 12479 .empty .empty)))) (.node 18461 (.node 14759 (.node 14087 (.node 13799 (.node 13109 .empty .empty) .empty) (.node 14411 .empty .empty)) (.node 16943 (.node 15467 (.node 15131 .empty .empty) .empty) (.node 17189 .empty .empty))) (.node 19961 (.node 19031 (.node 18749 (.node 18617 .empty .empty) .empty) (.node 19841 .empty .empty)) (.node 22853 (.node 21347 .empty .empty) (.node 23039 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2957_ordered : SortedNatTree.ordered row2957 = true := by decide
set_option maxRecDepth 4096 in
theorem row2957_values : SortedNatTree.values row2957 = Data.row2957 := rfl
theorem row2957_checked : SortedNatTree.contains row2957 = Data.row2957.contains := by
  rw [SortedNatTree.contains_eq_list row2957 row2957_ordered, row2957_values]
def row2963 : SortedNatTree.Tree := (.node 13103 (.node 6689 (.node 4691 (.node 4079 (.node 4019 (.node 3557 .empty .empty) .empty) (.node 4421 (.node 4391 .empty .empty) .empty)) (.node 5693 (.node 5381 (.node 4919 .empty .empty) .empty) (.node 6449 (.node 5861 .empty .empty) .empty))) (.node 8663 (.node 7607 (.node 7187 (.node 7151 .empty .empty) .empty) (.node 8111 (.node 7691 .empty .empty) .empty)) (.node 9851 (.node 9371 (.node 9011 .empty .empty) .empty) (.node 13049 (.node 11867 .empty .empty) .empty)))) (.node 16673 (.node 14303 (.node 13883 (.node 13679 (.node 13187 .empty .empty) .empty) (.node 14033 (.node 13931 .empty .empty) .empty)) (.node 15161 (.node 14939 (.node 14783 .empty .empty) .empty) (.node 16193 (.node 15629 .empty .empty) .empty))) (.node 19427 (.node 17627 (.node 17393 (.node 16871 .empty .empty) .empty) (.node 18251 (.node 18233 .empty .empty) .empty)) (.node 22541 (.node 21737 (.node 21401 .empty .empty) .empty) (.node 23057 (.node 22619 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2963_ordered : SortedNatTree.ordered row2963 = true := by decide
set_option maxRecDepth 4096 in
theorem row2963_values : SortedNatTree.values row2963 = Data.row2963 := rfl
theorem row2963_checked : SortedNatTree.contains row2963 = Data.row2963.contains := by
  rw [SortedNatTree.contains_eq_list row2963 row2963_ordered, row2963_values]
def row2969 : SortedNatTree.Tree := (.node 9281 (.node 7673 (.node 5009 (.node 4547 (.node 3557 (.node 3539 .empty .empty) .empty) (.node 4703 .empty .empty)) (.node 6701 (.node 5153 .empty .empty) (.node 6803 .empty .empty))) (.node 8807 (.node 7841 (.node 7823 .empty .empty) (.node 8597 .empty .empty)) (.node 9011 (.node 8861 .empty .empty) (.node 9041 .empty .empty)))) (.node 15173 (.node 12953 (.node 12197 (.node 10067 .empty .empty) (.node 12809 .empty .empty)) (.node 14561 (.node 13451 .empty .empty) (.node 15107 .empty .empty))) (.node 17987 (.node 16493 (.node 15383 .empty .empty) (.node 16661 .empty .empty)) (.node 21377 (.node 19937 .empty .empty) (.node 22037 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2969_ordered : SortedNatTree.ordered row2969 = true := by decide
set_option maxRecDepth 4096 in
theorem row2969_values : SortedNatTree.values row2969 = Data.row2969 := rfl
theorem row2969_checked : SortedNatTree.contains row2969 = Data.row2969.contains := by
  rw [SortedNatTree.contains_eq_list row2969 row2969_ordered, row2969_values]
def row2971 : SortedNatTree.Tree := (.node 12589 (.node 6871 (.node 4639 (.node 4057 (.node 3709 (.node 3037 .empty .empty) .empty) (.node 4273 .empty .empty)) (.node 5953 (.node 5557 .empty .empty) (.node 6067 .empty .empty))) (.node 9343 (.node 7573 (.node 7459 (.node 6961 .empty .empty) .empty) (.node 9337 .empty .empty)) (.node 9649 (.node 9511 .empty .empty) (.node 12421 .empty .empty)))) (.node 17713 (.node 15271 (.node 14221 (.node 13591 (.node 13477 .empty .empty) .empty) (.node 15061 .empty .empty)) (.node 17257 (.node 15649 .empty .empty) (.node 17419 .empty .empty))) (.node 21481 (.node 20011 (.node 18121 (.node 18049 .empty .empty) .empty) (.node 20353 .empty .empty)) (.node 22129 (.node 21859 .empty .empty) (.node 22963 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2971_ordered : SortedNatTree.ordered row2971 = true := by decide
set_option maxRecDepth 4096 in
theorem row2971_values : SortedNatTree.values row2971 = Data.row2971 := rfl
theorem row2971_checked : SortedNatTree.contains row2971 = Data.row2971.contains := by
  rw [SortedNatTree.contains_eq_list row2971 row2971_ordered, row2971_values]
def row2999 : SortedNatTree.Tree := (.node 11273 (.node 7349 (.node 4259 (.node 3881 (.node 3413 (.node 3371 .empty .empty) .empty) (.node 4073 .empty .empty)) (.node 5003 (.node 4871 (.node 4373 .empty .empty) .empty) (.node 6311 .empty .empty))) (.node 8951 (.node 7907 (.node 7577 (.node 7481 .empty .empty) .empty) (.node 8537 .empty .empty)) (.node 10799 (.node 10601 (.node 10463 .empty .empty) .empty) (.node 10979 .empty .empty)))) (.node 16607 (.node 14669 (.node 12653 (.node 11699 (.node 11483 .empty .empty) .empty) (.node 14033 .empty .empty)) (.node 16301 (.node 15299 (.node 14957 .empty .empty) .empty) (.node 16319 .empty .empty))) (.node 18713 (.node 17789 (.node 17609 (.node 17117 .empty .empty) .empty) (.node 18191 .empty .empty)) (.node 21569 (.node 19121 .empty .empty) (.node 22067 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2999_ordered : SortedNatTree.ordered row2999 = true := by decide
set_option maxRecDepth 4096 in
theorem row2999_values : SortedNatTree.values row2999 = Data.row2999 := rfl
theorem row2999_checked : SortedNatTree.contains row2999 = Data.row2999.contains := by
  rw [SortedNatTree.contains_eq_list row2999 row2999_ordered, row2999_values]
def row3001 : SortedNatTree.Tree := (.node 9931 (.node 5749 (.node 4021 (.node 3499 (.node 3391 (.node 3163 .empty .empty) .empty) (.node 3967 (.node 3727 .empty .empty) .empty)) (.node 4909 (.node 4591 (.node 4219 .empty .empty) .empty) (.node 4969 (.node 4951 .empty .empty) .empty))) (.node 7621 (.node 6733 (.node 6607 (.node 6121 .empty .empty) .empty) (.node 7573 (.node 7069 .empty .empty) .empty)) (.node 9007 (.node 8647 (.node 7789 .empty .empty) .empty) (.node 9697 .empty .empty)))) (.node 15289 (.node 11173 (.node 10687 (.node 10477 (.node 10267 .empty .empty) .empty) (.node 11059 (.node 10753 .empty .empty) .empty)) (.node 12517 (.node 12301 (.node 11941 .empty .empty) .empty) (.node 14341 .empty .empty))) (.node 19447 (.node 18367 (.node 17047 (.node 16981 .empty .empty) .empty) (.node 19381 (.node 19069 .empty .empty) .empty)) (.node 20323 (.node 20071 (.node 19777 .empty .empty) .empty) (.node 22741 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3001_ordered : SortedNatTree.ordered row3001 = true := by decide
set_option maxRecDepth 4096 in
theorem row3001_values : SortedNatTree.values row3001 = Data.row3001 := rfl
theorem row3001_checked : SortedNatTree.contains row3001 = Data.row3001.contains := by
  rw [SortedNatTree.contains_eq_list row3001 row3001_ordered, row3001_values]
def row3011 : SortedNatTree.Tree := (.node 9437 (.node 7127 (.node 4817 (.node 3329 (.node 3023 .empty .empty) (.node 4493 .empty .empty)) (.node 6353 (.node 6329 .empty .empty) (.node 6833 .empty .empty))) (.node 8009 (.node 7883 (.node 7349 .empty .empty) (.node 7937 .empty .empty)) (.node 8369 (.node 8147 .empty .empty) .empty))) (.node 19037 (.node 14207 (.node 11243 (.node 10151 .empty .empty) (.node 12923 .empty .empty)) (.node 17021 (.node 15791 .empty .empty) (.node 18899 .empty .empty))) (.node 21713 (.node 20747 (.node 20231 .empty .empty) (.node 21347 .empty .empty)) (.node 22079 (.node 21767 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3011_ordered : SortedNatTree.ordered row3011 = true := by decide
set_option maxRecDepth 4096 in
theorem row3011_values : SortedNatTree.values row3011 = Data.row3011 := rfl
theorem row3011_checked : SortedNatTree.contains row3011 = Data.row3011.contains := by
  rw [SortedNatTree.contains_eq_list row3011 row3011_ordered, row3011_values]
def row3019 : SortedNatTree.Tree := (.node 9181 (.node 6553 (.node 4621 (.node 4519 (.node 3571 (.node 3217 .empty .empty) .empty) (.node 4603 .empty .empty)) (.node 5779 (.node 5419 (.node 5413 .empty .empty) .empty) (.node 5839 .empty .empty))) (.node 7603 (.node 7351 (.node 6841 (.node 6571 .empty .empty) .empty) (.node 7417 .empty .empty)) (.node 8629 (.node 8317 (.node 8269 .empty .empty) .empty) (.node 8941 .empty .empty)))) (.node 13399 (.node 11839 (.node 10177 (.node 9613 (.node 9277 .empty .empty) .empty) (.node 11047 .empty .empty)) (.node 12487 (.node 12049 (.node 11887 .empty .empty) .empty) (.node 13099 .empty .empty))) (.node 19753 (.node 16063 (.node 15919 (.node 15139 .empty .empty) .empty) (.node 19483 .empty .empty)) (.node 22063 (.node 20599 (.node 19843 .empty .empty) .empty) (.node 22441 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3019_ordered : SortedNatTree.ordered row3019 = true := by decide
set_option maxRecDepth 4096 in
theorem row3019_values : SortedNatTree.values row3019 = Data.row3019 := rfl
theorem row3019_checked : SortedNatTree.contains row3019 = Data.row3019.contains := by
  rw [SortedNatTree.contains_eq_list row3019 row3019_ordered, row3019_values]
def row3023 : SortedNatTree.Tree := (.node 9497 (.node 6551 (.node 4637 (.node 3989 (.node 3767 (.node 3527 .empty .empty) .empty) (.node 4229 .empty .empty)) (.node 5879 (.node 4649 .empty .empty) (.node 5981 .empty .empty))) (.node 8669 (.node 7589 (.node 6959 (.node 6917 .empty .empty) .empty) (.node 8069 .empty .empty)) (.node 9059 (.node 8783 .empty .empty) (.node 9437 .empty .empty)))) (.node 16673 (.node 14057 (.node 11369 (.node 10223 (.node 10139 .empty .empty) .empty) (.node 11393 .empty .empty)) (.node 14543 (.node 14447 .empty .empty) (.node 15683 .empty .empty))) (.node 19001 (.node 17909 (.node 17627 .empty .empty) (.node 18131 .empty .empty)) (.node 20177 (.node 19961 .empty .empty) (.node 21893 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3023_ordered : SortedNatTree.ordered row3023 = true := by decide
set_option maxRecDepth 4096 in
theorem row3023_values : SortedNatTree.values row3023 = Data.row3023 := rfl
theorem row3023_checked : SortedNatTree.contains row3023 = Data.row3023.contains := by
  rw [SortedNatTree.contains_eq_list row3023 row3023_ordered, row3023_values]
def row3037 : SortedNatTree.Tree := (.node 10357 (.node 6301 (.node 4423 (.node 3433 (.node 3331 (.node 3253 .empty .empty) .empty) (.node 3709 (.node 3517 .empty .empty) .empty)) (.node 5689 (.node 4987 (.node 4507 .empty .empty) .empty) (.node 6007 (.node 5701 .empty .empty) .empty))) (.node 8461 (.node 7129 (.node 6823 (.node 6607 .empty .empty) .empty) (.node 8431 (.node 8329 .empty .empty) .empty)) (.node 9397 (.node 8713 (.node 8707 .empty .empty) .empty) (.node 9787 .empty .empty)))) (.node 17443 (.node 14557 (.node 11437 (.node 11299 (.node 11119 .empty .empty) .empty) (.node 12163 (.node 11971 .empty .empty) .empty)) (.node 15901 (.node 15877 (.node 14797 .empty .empty) .empty) (.node 17209 (.node 16057 .empty .empty) .empty))) (.node 20707 (.node 18313 (.node 18043 (.node 17539 .empty .empty) .empty) (.node 20089 (.node 19387 .empty .empty) .empty)) (.node 22039 (.node 21487 (.node 20887 .empty .empty) .empty) (.node 22651 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3037_ordered : SortedNatTree.ordered row3037 = true := by decide
set_option maxRecDepth 4096 in
theorem row3037_values : SortedNatTree.values row3037 = Data.row3037 := rfl
theorem row3037_checked : SortedNatTree.contains row3037 = Data.row3037.contains := by
  rw [SortedNatTree.contains_eq_list row3037 row3037_ordered, row3037_values]
def row3041 : SortedNatTree.Tree := (.node 11987 (.node 6719 (.node 4337 (.node 3533 (.node 3491 (.node 3209 .empty .empty) .empty) (.node 4133 .empty .empty)) (.node 5897 (.node 4409 .empty .empty) (.node 6317 .empty .empty))) (.node 9839 (.node 8273 (.node 7529 (.node 7481 .empty .empty) .empty) (.node 8963 .empty .empty)) (.node 10601 (.node 10133 .empty .empty) (.node 11903 .empty .empty)))) (.node 16487 (.node 15473 (.node 13001 (.node 12659 (.node 12323 .empty .empty) .empty) (.node 14843 .empty .empty)) (.node 15761 (.node 15671 .empty .empty) (.node 16481 .empty .empty))) (.node 18671 (.node 17729 (.node 16763 .empty .empty) (.node 17909 .empty .empty)) (.node 20759 (.node 20249 .empty .empty) (.node 21821 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3041_ordered : SortedNatTree.ordered row3041 = true := by decide
set_option maxRecDepth 4096 in
theorem row3041_values : SortedNatTree.values row3041 = Data.row3041 := rfl
theorem row3041_checked : SortedNatTree.contains row3041 = Data.row3041.contains := by
  rw [SortedNatTree.contains_eq_list row3041 row3041_ordered, row3041_values]
def row3049 : SortedNatTree.Tree := (.node 11743 (.node 8191 (.node 5581 (.node 4831 (.node 4153 (.node 3259 .empty .empty) .empty) (.node 5167 .empty .empty)) (.node 7297 (.node 7177 (.node 6379 .empty .empty) .empty) (.node 8089 .empty .empty))) (.node 9871 (.node 9787 (.node 8737 (.node 8563 .empty .empty) .empty) (.node 9811 .empty .empty)) (.node 10357 (.node 9949 .empty .empty) (.node 10663 .empty .empty)))) (.node 18253 (.node 16111 (.node 13723 (.node 12889 (.node 12007 .empty .empty) .empty) (.node 15031 .empty .empty)) (.node 17791 (.node 17761 (.node 16657 .empty .empty) .empty) (.node 18217 .empty .empty))) (.node 20899 (.node 20233 (.node 19819 (.node 18637 .empty .empty) .empty) (.node 20323 .empty .empty)) (.node 21187 (.node 20947 .empty .empty) (.node 21247 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3049_ordered : SortedNatTree.ordered row3049 = true := by decide
set_option maxRecDepth 4096 in
theorem row3049_values : SortedNatTree.values row3049 = Data.row3049 := rfl
theorem row3049_checked : SortedNatTree.contains row3049 = Data.row3049.contains := by
  rw [SortedNatTree.contains_eq_list row3049 row3049_ordered, row3049_values]
def row3061 : SortedNatTree.Tree := (.node 12853 (.node 6829 (.node 5437 (.node 3613 (.node 3607 (.node 3571 .empty .empty) .empty) (.node 4729 .empty .empty)) (.node 6343 (.node 5857 .empty .empty) (.node 6673 .empty .empty))) (.node 8719 (.node 8293 (.node 8167 (.node 6907 .empty .empty) .empty) (.node 8563 .empty .empty)) (.node 12619 (.node 12511 .empty .empty) (.node 12721 .empty .empty)))) (.node 16843 (.node 15877 (.node 15259 (.node 13963 (.node 13627 .empty .empty) .empty) (.node 15493 .empty .empty)) (.node 16333 (.node 15919 .empty .empty) (.node 16567 .empty .empty))) (.node 21433 (.node 20011 (.node 17491 .empty .empty) (.node 21391 .empty .empty)) (.node 22717 (.node 22093 .empty .empty) (.node 22741 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3061_ordered : SortedNatTree.ordered row3061 = true := by decide
set_option maxRecDepth 4096 in
theorem row3061_values : SortedNatTree.values row3061 = Data.row3061 := rfl
theorem row3061_checked : SortedNatTree.contains row3061 = Data.row3061.contains := by
  rw [SortedNatTree.contains_eq_list row3061 row3061_ordered, row3061_values]
def row3067 : SortedNatTree.Tree := (.node 10567 (.node 7687 (.node 5419 (.node 5167 (.node 4909 (.node 3373 .empty .empty) .empty) (.node 5413 .empty .empty)) (.node 7459 (.node 6661 (.node 5737 .empty .empty) .empty) (.node 7489 .empty .empty))) (.node 9127 (.node 8929 (.node 8269 (.node 7753 .empty .empty) .empty) (.node 9001 .empty .empty)) (.node 10069 (.node 9829 (.node 9787 .empty .empty) .empty) (.node 10399 .empty .empty)))) (.node 17053 (.node 12547 (.node 11941 (.node 11491 (.node 10909 .empty .empty) .empty) (.node 11953 .empty .empty)) (.node 15667 (.node 15439 (.node 12823 .empty .empty) .empty) (.node 15901 .empty .empty))) (.node 21517 (.node 20551 (.node 20509 (.node 17683 .empty .empty) .empty) (.node 21391 .empty .empty)) (.node 22303 (.node 21787 (.node 21757 .empty .empty) .empty) (.node 22441 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3067_ordered : SortedNatTree.ordered row3067 = true := by decide
set_option maxRecDepth 4096 in
theorem row3067_values : SortedNatTree.values row3067 = Data.row3067 := rfl
theorem row3067_checked : SortedNatTree.contains row3067 = Data.row3067.contains := by
  rw [SortedNatTree.contains_eq_list row3067 row3067_ordered, row3067_values]
def row3079 : SortedNatTree.Tree := (.node 9679 (.node 6679 (.node 4273 (.node 3709 (.node 3673 (.node 3181 .empty .empty) .empty) (.node 3877 .empty .empty)) (.node 5641 (.node 4957 (.node 4591 .empty .empty) .empty) (.node 6301 .empty .empty))) (.node 7867 (.node 7561 (.node 7297 (.node 6709 .empty .empty) .empty) (.node 7699 .empty .empty)) (.node 8821 (.node 8329 (.node 8161 .empty .empty) .empty) (.node 9463 .empty .empty)))) (.node 16363 (.node 11689 (.node 10597 (.node 10399 (.node 9973 .empty .empty) .empty) (.node 11047 .empty .empty)) (.node 14281 (.node 13633 (.node 12037 .empty .empty) .empty) (.node 14419 .empty .empty))) (.node 20047 (.node 18493 (.node 17497 (.node 17419 .empty .empty) .empty) (.node 18553 .empty .empty)) (.node 21379 (.node 20287 .empty .empty) (.node 21841 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3079_ordered : SortedNatTree.ordered row3079 = true := by decide
set_option maxRecDepth 4096 in
theorem row3079_values : SortedNatTree.values row3079 = Data.row3079 := rfl
theorem row3079_checked : SortedNatTree.contains row3079 = Data.row3079.contains := by
  rw [SortedNatTree.contains_eq_list row3079 row3079_ordered, row3079_values]
def row3083 : SortedNatTree.Tree := (.node 10133 (.node 6257 (.node 4703 (.node 3797 (.node 3449 (.node 3329 .empty .empty) .empty) (.node 4373 .empty .empty)) (.node 4931 (.node 4871 .empty .empty) (.node 5333 .empty .empty))) (.node 8747 (.node 7307 (.node 7187 (.node 6737 .empty .empty) .empty) (.node 7907 .empty .empty)) (.node 9461 (.node 8963 .empty .empty) (.node 9929 .empty .empty)))) (.node 17939 (.node 15053 (.node 12197 (.node 11699 (.node 11519 .empty .empty) .empty) (.node 14411 .empty .empty)) (.node 17021 (.node 15467 .empty .empty) (.node 17903 .empty .empty))) (.node 21347 (.node 20759 (.node 18797 (.node 18251 .empty .empty) .empty) (.node 21149 .empty .empty)) (.node 22031 (.node 21803 .empty .empty) (.node 22691 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3083_ordered : SortedNatTree.ordered row3083 = true := by decide
set_option maxRecDepth 4096 in
theorem row3083_values : SortedNatTree.values row3083 = Data.row3083 := rfl
theorem row3083_checked : SortedNatTree.contains row3083 = Data.row3083.contains := by
  rw [SortedNatTree.contains_eq_list row3083 row3083_ordered, row3083_values]
def row3089 : SortedNatTree.Tree := (.node 11579 (.node 6689 (.node 5147 (.node 3851 (.node 3299 .empty .empty) (.node 3917 .empty .empty)) (.node 5639 (.node 5303 .empty .empty) (.node 5867 .empty .empty))) (.node 9203 (.node 7541 (.node 7457 .empty .empty) (.node 8093 .empty .empty)) (.node 10169 (.node 9629 .empty .empty) (.node 11279 .empty .empty)))) (.node 19973 (.node 16889 (.node 14741 (.node 11789 .empty .empty) (.node 15287 .empty .empty)) (.node 18341 (.node 18257 .empty .empty) (.node 18803 .empty .empty))) (.node 22433 (.node 21611 (.node 20327 .empty .empty) (.node 22091 .empty .empty)) (.node 22619 (.node 22511 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3089_ordered : SortedNatTree.ordered row3089 = true := by decide
set_option maxRecDepth 4096 in
theorem row3089_values : SortedNatTree.values row3089 = Data.row3089 := rfl
theorem row3089_checked : SortedNatTree.contains row3089 = Data.row3089.contains := by
  rw [SortedNatTree.contains_eq_list row3089 row3089_ordered, row3089_values]
def row3109 : SortedNatTree.Tree := (.node 9949 (.node 6733 (.node 4447 (.node 4057 (.node 3877 (.node 3541 .empty .empty) (.node 4027 .empty .empty)) (.node 4339 (.node 4099 .empty .empty) .empty)) (.node 5791 (.node 4801 (.node 4759 .empty .empty) .empty) (.node 6607 (.node 6073 .empty .empty) .empty))) (.node 8677 (.node 7177 (.node 6997 (.node 6883 .empty .empty) (.node 7069 .empty .empty)) (.node 7993 (.node 7603 .empty .empty) .empty)) (.node 9697 (.node 9157 (.node 8941 .empty .empty) .empty) (.node 9883 (.node 9781 .empty .empty) .empty)))) (.node 15511 (.node 12391 (.node 12109 (.node 10903 (.node 10459 .empty .empty) (.node 11353 .empty .empty)) (.node 12373 (.node 12211 .empty .empty) .empty)) (.node 14251 (.node 14083 (.node 13597 .empty .empty) .empty) (.node 15271 (.node 14563 .empty .empty) .empty))) (.node 18217 (.node 17707 (.node 16993 (.node 15739 .empty .empty) (.node 17137 .empty .empty)) (.node 17863 (.node 17713 .empty .empty) .empty)) (.node 21937 (.node 21277 (.node 21001 .empty .empty) .empty) (.node 22093 (.node 21997 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row3109_ordered : SortedNatTree.ordered row3109 = true := by decide
set_option maxRecDepth 4096 in
theorem row3109_values : SortedNatTree.values row3109 = Data.row3109 := rfl
theorem row3109_checked : SortedNatTree.contains row3109 = Data.row3109.contains := by
  rw [SortedNatTree.contains_eq_list row3109 row3109_ordered, row3109_values]
def row3119 : SortedNatTree.Tree := (.node 12647 (.node 6857 (.node 5741 (.node 4679 (.node 4391 (.node 3677 .empty .empty) .empty) (.node 5237 (.node 5231 .empty .empty) .empty)) (.node 6311 (.node 6029 (.node 5783 .empty .empty) .empty) (.node 6491 (.node 6359 .empty .empty) .empty))) (.node 10139 (.node 9227 (.node 8831 (.node 7109 .empty .empty) .empty) (.node 10133 (.node 9887 .empty .empty) .empty)) (.node 11471 (.node 10853 (.node 10601 .empty .empty) .empty) (.node 11681 .empty .empty)))) (.node 17747 (.node 15467 (.node 13913 (.node 13907 (.node 13757 .empty .empty) .empty) (.node 14879 (.node 14087 .empty .empty) .empty)) (.node 16763 (.node 16433 (.node 16223 .empty .empty) .empty) (.node 17609 .empty .empty))) (.node 21569 (.node 20051 (.node 19919 (.node 19373 .empty .empty) .empty) (.node 21341 (.node 21227 .empty .empty) .empty)) (.node 22367 (.node 22283 (.node 21683 .empty .empty) .empty) (.node 22697 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3119_ordered : SortedNatTree.ordered row3119 = true := by decide
set_option maxRecDepth 4096 in
theorem row3119_values : SortedNatTree.values row3119 = Data.row3119 := rfl
theorem row3119_checked : SortedNatTree.contains row3119 = Data.row3119.contains := by
  rw [SortedNatTree.contains_eq_list row3119 row3119_ordered, row3119_values]
def row3121 : SortedNatTree.Tree := (.node 10993 (.node 5851 (.node 4357 (.node 4201 (.node 3673 (.node 3433 .empty .empty) .empty) (.node 4219 .empty .empty)) (.node 4639 (.node 4597 (.node 4507 .empty .empty) .empty) (.node 4903 .empty .empty))) (.node 8599 (.node 7759 (.node 7507 (.node 7393 .empty .empty) .empty) (.node 8221 .empty .empty)) (.node 10639 (.node 9187 (.node 9043 .empty .empty) .empty) (.node 10657 .empty .empty)))) (.node 18379 (.node 14563 (.node 11887 (.node 11677 (.node 11497 .empty .empty) .empty) (.node 13903 .empty .empty)) (.node 16759 (.node 15373 (.node 14731 .empty .empty) .empty) (.node 16843 .empty .empty))) (.node 19081 (.node 18973 (.node 18913 (.node 18427 .empty .empty) .empty) (.node 19009 .empty .empty)) (.node 21787 (.node 21319 .empty .empty) (.node 22483 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3121_ordered : SortedNatTree.ordered row3121 = true := by decide
set_option maxRecDepth 4096 in
theorem row3121_values : SortedNatTree.values row3121 = Data.row3121 := rfl
theorem row3121_checked : SortedNatTree.contains row3121 = Data.row3121.contains := by
  rw [SortedNatTree.contains_eq_list row3121 row3121_ordered, row3121_values]
def row3137 : SortedNatTree.Tree := (.node 10601 (.node 6803 (.node 5009 (.node 3821 (.node 3593 (.node 3371 .empty .empty) .empty) (.node 4349 .empty .empty)) (.node 5501 (.node 5189 .empty .empty) (.node 6311 .empty .empty))) (.node 9767 (.node 8387 (.node 7919 (.node 7331 .empty .empty) .empty) (.node 8513 .empty .empty)) (.node 10079 (.node 9923 .empty .empty) (.node 10427 .empty .empty)))) (.node 14867 (.node 12107 (.node 10889 (.node 10781 (.node 10667 .empty .empty) .empty) (.node 10937 .empty .empty)) (.node 13931 (.node 13217 .empty .empty) (.node 14081 .empty .empty))) (.node 20663 (.node 18371 (.node 17387 (.node 17333 .empty .empty) .empty) (.node 18803 .empty .empty)) (.node 22067 (.node 20681 .empty .empty) (.node 22469 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3137_ordered : SortedNatTree.ordered row3137 = true := by decide
set_option maxRecDepth 4096 in
theorem row3137_values : SortedNatTree.values row3137 = Data.row3137 := rfl
theorem row3137_checked : SortedNatTree.contains row3137 = Data.row3137.contains := by
  rw [SortedNatTree.contains_eq_list row3137 row3137_ordered, row3137_values]
def row3163 : SortedNatTree.Tree := (.node 13537 (.node 8527 (.node 5431 (.node 3967 (.node 3571 (.node 3529 .empty .empty) (.node 3769 .empty .empty)) (.node 5101 (.node 4129 .empty .empty) .empty)) (.node 6733 (.node 6037 (.node 5743 .empty .empty) (.node 6301 .empty .empty)) (.node 7951 (.node 7477 .empty .empty) .empty))) (.node 10009 (.node 9187 (.node 8821 (.node 8707 .empty .empty) (.node 9043 .empty .empty)) (.node 9859 (.node 9721 .empty .empty) .empty)) (.node 11329 (.node 10711 (.node 10333 .empty .empty) (.node 10861 .empty .empty)) (.node 12967 (.node 12277 .empty .empty) .empty)))) (.node 18481 (.node 15679 (.node 14431 (.node 13873 (.node 13597 .empty .empty) (.node 14143 .empty .empty)) (.node 15541 (.node 14851 .empty .empty) .empty)) (.node 17923 (.node 17389 (.node 16993 .empty .empty) (.node 17401 .empty .empty)) (.node 18049 (.node 17977 .empty .empty) .empty))) (.node 21157 (.node 20287 (.node 18553 (.node 18523 .empty .empty) (.node 19891 .empty .empty)) (.node 20731 (.node 20611 .empty .empty) .empty)) (.node 21871 (.node 21727 (.node 21391 .empty .empty) (.node 21787 .empty .empty)) (.node 22861 (.node 22483 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row3163_ordered : SortedNatTree.ordered row3163 = true := by decide
set_option maxRecDepth 4096 in
theorem row3163_values : SortedNatTree.values row3163 = Data.row3163 := rfl
theorem row3163_checked : SortedNatTree.contains row3163 = Data.row3163.contains := by
  rw [SortedNatTree.contains_eq_list row3163 row3163_ordered, row3163_values]
def row3167 : SortedNatTree.Tree := (.node 10859 (.node 6299 (.node 4421 (.node 4001 (.node 3527 .empty .empty) (.node 4217 .empty .empty)) (.node 5711 (.node 5309 .empty .empty) (.node 6269 .empty .empty))) (.node 9239 (.node 8039 (.node 6521 .empty .empty) (.node 8951 .empty .empty)) (.node 9413 (.node 9377 .empty .empty) (.node 10463 .empty .empty)))) (.node 16223 (.node 12917 (.node 12113 (.node 11657 .empty .empty) (.node 12269 .empty .empty)) (.node 14753 (.node 13967 .empty .empty) (.node 14831 .empty .empty))) (.node 19289 (.node 16979 (.node 16811 .empty .empty) (.node 17189 .empty .empty)) (.node 20753 (.node 19421 .empty .empty) (.node 22619 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3167_ordered : SortedNatTree.ordered row3167 = true := by decide
set_option maxRecDepth 4096 in
theorem row3167_values : SortedNatTree.values row3167 = Data.row3167 := rfl
theorem row3167_checked : SortedNatTree.contains row3167 = Data.row3167.contains := by
  rw [SortedNatTree.contains_eq_list row3167 row3167_ordered, row3167_values]
def row3169 : SortedNatTree.Tree := (.node 11149 (.node 6361 (.node 5101 (.node 3391 (.node 3229 .empty .empty) (.node 3733 .empty .empty)) (.node 5953 (.node 5119 .empty .empty) (.node 6037 .empty .empty))) (.node 8419 (.node 8179 (.node 7213 .empty .empty) (.node 8377 .empty .empty)) (.node 10639 (.node 10369 .empty .empty) .empty))) (.node 15541 (.node 13873 (.node 12697 (.node 12049 .empty .empty) (.node 12823 .empty .empty)) (.node 15493 (.node 14503 .empty .empty) .empty)) (.node 19717 (.node 17683 (.node 15559 .empty .empty) (.node 18493 .empty .empty)) (.node 22651 (.node 22051 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3169_ordered : SortedNatTree.ordered row3169 = true := by decide
set_option maxRecDepth 4096 in
theorem row3169_values : SortedNatTree.values row3169 = Data.row3169 := rfl
theorem row3169_checked : SortedNatTree.contains row3169 = Data.row3169.contains := by
  rw [SortedNatTree.contains_eq_list row3169 row3169_ordered, row3169_values]
def row3181 : SortedNatTree.Tree := (.node 10729 (.node 6073 (.node 4507 (.node 4219 (.node 4099 (.node 3343 .empty .empty) .empty) (.node 4273 .empty .empty)) (.node 5107 (.node 5011 (.node 4903 .empty .empty) .empty) (.node 5527 .empty .empty))) (.node 9043 (.node 7507 (.node 6709 (.node 6481 .empty .empty) .empty) (.node 8377 .empty .empty)) (.node 9421 (.node 9127 .empty .empty) (.node 9973 .empty .empty)))) (.node 17383 (.node 13687 (.node 11587 (.node 11197 (.node 11113 .empty .empty) .empty) (.node 12253 .empty .empty)) (.node 15451 (.node 15361 .empty .empty) (.node 16063 .empty .empty))) (.node 20611 (.node 19267 (.node 18517 (.node 18217 .empty .empty) .empty) (.node 19687 .empty .empty)) (.node 22027 (.node 21871 .empty .empty) (.node 22543 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3181_ordered : SortedNatTree.ordered row3181 = true := by decide
set_option maxRecDepth 4096 in
theorem row3181_values : SortedNatTree.values row3181 = Data.row3181 := rfl
theorem row3181_checked : SortedNatTree.contains row3181 = Data.row3181.contains := by
  rw [SortedNatTree.contains_eq_list row3181 row3181_ordered, row3181_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
