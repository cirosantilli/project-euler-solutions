import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row3187 : SortedNatTree.Tree := (.node 10159 (.node 7411 (.node 4957 (.node 4561 (.node 3529 (.node 3229 .empty .empty) (.node 4093 .empty .empty)) (.node 4729 (.node 4657 .empty .empty) .empty)) (.node 5479 (.node 5227 (.node 5209 .empty .empty) .empty) (.node 6907 (.node 6379 .empty .empty) .empty))) (.node 8563 (.node 8101 (.node 8017 (.node 7477 .empty .empty) .empty) (.node 8527 (.node 8233 .empty .empty) .empty)) (.node 9151 (.node 9109 (.node 9049 .empty .empty) .empty) (.node 9643 (.node 9391 .empty .empty) .empty)))) (.node 19207 (.node 15259 (.node 12637 (.node 11587 (.node 11353 .empty .empty) (.node 11593 .empty .empty)) (.node 14947 (.node 13147 .empty .empty) .empty)) (.node 17341 (.node 17299 (.node 17137 .empty .empty) .empty) (.node 18181 (.node 17551 .empty .empty) .empty))) (.node 20173 (.node 19543 (.node 19447 (.node 19273 .empty .empty) .empty) (.node 20161 (.node 20071 .empty .empty) .empty)) (.node 21019 (.node 20887 (.node 20599 .empty .empty) .empty) (.node 22153 (.node 21817 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row3187_ordered : SortedNatTree.ordered row3187 = true := by decide
set_option maxRecDepth 4096 in
theorem row3187_values : SortedNatTree.values row3187 = Data.row3187 := rfl
theorem row3187_checked : SortedNatTree.contains row3187 = Data.row3187.contains := by
  rw [SortedNatTree.contains_eq_list row3187 row3187_ordered, row3187_values]
def row3191 : SortedNatTree.Tree := (.node 9161 (.node 7451 (.node 4583 (.node 3821 (.node 3221 .empty .empty) (.node 4397 .empty .empty)) (.node 6779 (.node 5531 .empty .empty) (.node 7043 .empty .empty))) (.node 7937 (.node 7703 (.node 7547 .empty .empty) (.node 7883 .empty .empty)) (.node 8111 (.node 8069 .empty .empty) (.node 8597 .empty .empty)))) (.node 14423 (.node 11927 (.node 10499 (.node 9539 .empty .empty) (.node 11831 .empty .empty)) (.node 13499 (.node 13217 .empty .empty) (.node 13931 .empty .empty))) (.node 17891 (.node 15413 (.node 15287 .empty .empty) (.node 17333 .empty .empty)) (.node 20297 (.node 18749 .empty .empty) (.node 20411 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3191_ordered : SortedNatTree.ordered row3191 = true := by decide
set_option maxRecDepth 4096 in
theorem row3191_values : SortedNatTree.values row3191 = Data.row3191 := rfl
theorem row3191_checked : SortedNatTree.contains row3191 = Data.row3191.contains := by
  rw [SortedNatTree.contains_eq_list row3191 row3191_ordered, row3191_values]
def row3203 : SortedNatTree.Tree := (.node 11597 (.node 8597 (.node 6827 (.node 5279 (.node 4481 (.node 3467 .empty .empty) .empty) (.node 6101 (.node 5867 .empty .empty) .empty)) (.node 7949 (.node 7541 (.node 7331 .empty .empty) .empty) (.node 8219 .empty .empty))) (.node 9311 (.node 9209 (.node 8933 (.node 8627 .empty .empty) .empty) (.node 9221 .empty .empty)) (.node 10223 (.node 10061 (.node 9923 .empty .empty) .empty) (.node 10253 .empty .empty)))) (.node 19037 (.node 16103 (.node 13619 (.node 13043 (.node 11903 .empty .empty) .empty) (.node 15263 .empty .empty)) (.node 18749 (.node 17483 (.node 17117 .empty .empty) .empty) (.node 18839 .empty .empty))) (.node 20123 (.node 19463 (.node 19163 (.node 19079 .empty .empty) .empty) (.node 19751 .empty .empty)) (.node 21059 (.node 21011 (.node 20771 .empty .empty) .empty) (.node 22811 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3203_ordered : SortedNatTree.ordered row3203 = true := by decide
set_option maxRecDepth 4096 in
theorem row3203_values : SortedNatTree.values row3203 = Data.row3203 := rfl
theorem row3203_checked : SortedNatTree.contains row3203 = Data.row3203.contains := by
  rw [SortedNatTree.contains_eq_list row3203 row3203_ordered, row3203_values]
def row3209 : SortedNatTree.Tree := (.node 11519 (.node 7433 (.node 5471 (.node 4721 (.node 3833 (.node 3617 .empty .empty) .empty) (.node 5417 (.node 5171 .empty .empty) .empty)) (.node 6761 (.node 6653 (.node 5717 .empty .empty) .empty) (.node 7187 .empty .empty))) (.node 9257 (.node 8699 (.node 7817 (.node 7523 .empty .empty) .empty) (.node 8861 .empty .empty)) (.node 10979 (.node 9497 (.node 9293 .empty .empty) .empty) (.node 11447 .empty .empty)))) (.node 18119 (.node 14051 (.node 13613 (.node 12821 (.node 12539 .empty .empty) .empty) (.node 13757 .empty .empty)) (.node 17159 (.node 16487 (.node 16067 .empty .empty) .empty) (.node 18077 .empty .empty))) (.node 21011 (.node 19889 (.node 19301 (.node 19163 .empty .empty) .empty) (.node 20681 .empty .empty)) (.node 22229 (.node 21929 (.node 21383 .empty .empty) .empty) (.node 22739 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3209_ordered : SortedNatTree.ordered row3209 = true := by decide
set_option maxRecDepth 4096 in
theorem row3209_values : SortedNatTree.values row3209 = Data.row3209 := rfl
theorem row3209_checked : SortedNatTree.contains row3209 = Data.row3209.contains := by
  rw [SortedNatTree.contains_eq_list row3209 row3209_ordered, row3209_values]
def row3217 : SortedNatTree.Tree := (.node 13759 (.node 7177 (.node 6073 (.node 4789 (.node 3931 (.node 3511 .empty .empty) .empty) (.node 5821 .empty .empty)) (.node 6301 (.node 6211 .empty .empty) (.node 6619 .empty .empty))) (.node 11197 (.node 10273 (.node 9907 .empty .empty) (.node 10861 .empty .empty)) (.node 12763 (.node 12421 .empty .empty) (.node 13417 .empty .empty)))) (.node 17581 (.node 15907 (.node 15649 (.node 15601 (.node 14737 .empty .empty) .empty) (.node 15739 .empty .empty)) (.node 16831 (.node 16063 .empty .empty) (.node 17377 .empty .empty))) (.node 20011 (.node 19483 (.node 17761 .empty .empty) (.node 19681 .empty .empty)) (.node 20743 (.node 20731 .empty .empty) (.node 22669 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3217_ordered : SortedNatTree.ordered row3217 = true := by decide
set_option maxRecDepth 4096 in
theorem row3217_values : SortedNatTree.values row3217 = Data.row3217 := rfl
theorem row3217_checked : SortedNatTree.contains row3217 = Data.row3217.contains := by
  rw [SortedNatTree.contains_eq_list row3217 row3217_ordered, row3217_values]
def row3221 : SortedNatTree.Tree := (.node 13931 (.node 6311 (.node 4409 (.node 3851 (.node 3767 (.node 3761 .empty .empty) .empty) (.node 4349 .empty .empty)) (.node 4817 (.node 4547 .empty .empty) (.node 6269 .empty .empty))) (.node 8609 (.node 7727 (.node 7643 (.node 7247 .empty .empty) .empty) (.node 8429 .empty .empty)) (.node 12239 (.node 10259 .empty .empty) (.node 13877 .empty .empty)))) (.node 18743 (.node 16631 (.node 15641 (.node 15467 (.node 14447 .empty .empty) .empty) (.node 16547 .empty .empty)) (.node 18143 (.node 17669 .empty .empty) (.node 18521 .empty .empty))) (.node 19841 (.node 18959 (.node 18803 .empty .empty) (.node 19697 .empty .empty)) (.node 21143 (.node 20849 .empty .empty) (.node 22247 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3221_ordered : SortedNatTree.ordered row3221 = true := by decide
set_option maxRecDepth 4096 in
theorem row3221_values : SortedNatTree.values row3221 = Data.row3221 := rfl
theorem row3221_checked : SortedNatTree.contains row3221 = Data.row3221.contains := by
  rw [SortedNatTree.contains_eq_list row3221 row3221_ordered, row3221_values]
def row3229 : SortedNatTree.Tree := (.node 9277 (.node 6553 (.node 4789 (.node 4057 (.node 3529 (.node 3463 .empty .empty) .empty) (.node 4621 (.node 4153 .empty .empty) .empty)) (.node 5581 (.node 5023 (.node 4993 .empty .empty) .empty) (.node 6007 .empty .empty))) (.node 8191 (.node 6841 (.node 6703 (.node 6673 .empty .empty) .empty) (.node 6997 (.node 6883 .empty .empty) .empty)) (.node 9109 (.node 8527 (.node 8293 .empty .empty) .empty) (.node 9151 .empty .empty)))) (.node 17911 (.node 14929 (.node 12451 (.node 9721 (.node 9511 .empty .empty) .empty) (.node 14923 (.node 12829 .empty .empty) .empty)) (.node 17467 (.node 17317 (.node 16927 .empty .empty) .empty) (.node 17509 .empty .empty))) (.node 20929 (.node 19417 (.node 18517 (.node 18307 .empty .empty) .empty) (.node 20743 (.node 20089 .empty .empty) .empty)) (.node 21649 (.node 21601 (.node 21019 .empty .empty) .empty) (.node 21727 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3229_ordered : SortedNatTree.ordered row3229 = true := by decide
set_option maxRecDepth 4096 in
theorem row3229_values : SortedNatTree.values row3229 = Data.row3229 := rfl
theorem row3229_checked : SortedNatTree.contains row3229 = Data.row3229.contains := by
  rw [SortedNatTree.contains_eq_list row3229 row3229_ordered, row3229_values]
def row3251 : SortedNatTree.Tree := (.node 13163 (.node 5039 (.node 4409 (.node 4241 (.node 3413 .empty .empty) (.node 4253 .empty .empty)) (.node 4973 (.node 4721 .empty .empty) .empty)) (.node 9413 (.node 8273 (.node 7229 .empty .empty) (.node 8963 .empty .empty)) (.node 11867 (.node 9497 .empty .empty) .empty))) (.node 17183 (.node 15629 (.node 14033 (.node 13907 .empty .empty) (.node 14831 .empty .empty)) (.node 16067 (.node 15671 .empty .empty) .empty)) (.node 19919 (.node 19031 (.node 17321 .empty .empty) (.node 19427 .empty .empty)) (.node 22409 (.node 20183 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3251_ordered : SortedNatTree.ordered row3251 = true := by decide
set_option maxRecDepth 4096 in
theorem row3251_values : SortedNatTree.values row3251 = Data.row3251 := rfl
theorem row3251_checked : SortedNatTree.contains row3251 = Data.row3251.contains := by
  rw [SortedNatTree.contains_eq_list row3251 row3251_ordered, row3251_values]
def row3253 : SortedNatTree.Tree := (.node 9967 (.node 5851 (.node 4339 (.node 3607 (.node 3547 (.node 3517 .empty .empty) .empty) (.node 4231 (.node 4219 .empty .empty) .empty)) (.node 5479 (.node 5197 (.node 4357 .empty .empty) .empty) (.node 5557 .empty .empty))) (.node 7927 (.node 6571 (.node 6067 (.node 6043 .empty .empty) .empty) (.node 7873 (.node 7717 .empty .empty) .empty)) (.node 8893 (.node 8641 (.node 8263 .empty .empty) .empty) (.node 9817 .empty .empty)))) (.node 16567 (.node 13033 (.node 12049 (.node 10501 (.node 9973 .empty .empty) .empty) (.node 12577 (.node 12073 .empty .empty) .empty)) (.node 14503 (.node 14293 (.node 13249 .empty .empty) .empty) (.node 15121 .empty .empty))) (.node 19717 (.node 18289 (.node 17257 (.node 16603 .empty .empty) .empty) (.node 18691 (.node 18493 .empty .empty) .empty)) (.node 20719 (.node 20389 (.node 20161 .empty .empty) .empty) (.node 22027 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3253_ordered : SortedNatTree.ordered row3253 = true := by decide
set_option maxRecDepth 4096 in
theorem row3253_values : SortedNatTree.values row3253 = Data.row3253 := rfl
theorem row3253_checked : SortedNatTree.contains row3253 = Data.row3253.contains := by
  rw [SortedNatTree.contains_eq_list row3253 row3253_ordered, row3253_values]
def row3257 : SortedNatTree.Tree := (.node 11069 (.node 6053 (.node 4937 (.node 4547 (.node 3449 (.node 3407 .empty .empty) .empty) (.node 4649 .empty .empty)) (.node 5471 (.node 5441 (.node 5051 .empty .empty) .empty) (.node 5507 .empty .empty))) (.node 9059 (.node 6917 (.node 6491 (.node 6221 .empty .empty) .empty) (.node 7949 .empty .empty)) (.node 10709 (.node 9587 .empty .empty) (.node 11003 .empty .empty)))) (.node 17417 (.node 12923 (.node 12251 (.node 11807 (.node 11261 .empty .empty) .empty) (.node 12347 .empty .empty)) (.node 13151 (.node 13127 .empty .empty) (.node 13679 .empty .empty))) (.node 18713 (.node 18191 (.node 18089 (.node 17519 .empty .empty) .empty) (.node 18257 .empty .empty)) (.node 20939 (.node 20333 .empty .empty) (.node 21191 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3257_ordered : SortedNatTree.ordered row3257 = true := by decide
set_option maxRecDepth 4096 in
theorem row3257_values : SortedNatTree.values row3257 = Data.row3257 := rfl
theorem row3257_checked : SortedNatTree.contains row3257 = Data.row3257.contains := by
  rw [SortedNatTree.contains_eq_list row3257 row3257_ordered, row3257_values]
def row3259 : SortedNatTree.Tree := (.node 10771 (.node 6199 (.node 4567 (.node 4021 (.node 3271 .empty .empty) (.node 4327 .empty .empty)) (.node 5503 (.node 5107 .empty .empty) (.node 5569 .empty .empty))) (.node 7963 (.node 7207 (.node 6247 .empty .empty) (.node 7717 .empty .empty)) (.node 8689 (.node 8641 .empty .empty) (.node 9067 .empty .empty)))) (.node 18013 (.node 12583 (.node 11953 (.node 11527 .empty .empty) (.node 12343 .empty .empty)) (.node 15787 (.node 12721 .empty .empty) (.node 17467 .empty .empty))) (.node 20731 (.node 19219 (.node 18757 .empty .empty) (.node 19387 .empty .empty)) (.node 21121 (.node 20857 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3259_ordered : SortedNatTree.ordered row3259 = true := by decide
set_option maxRecDepth 4096 in
theorem row3259_values : SortedNatTree.values row3259 = Data.row3259 := rfl
theorem row3259_checked : SortedNatTree.contains row3259 = Data.row3259.contains := by
  rw [SortedNatTree.contains_eq_list row3259 row3259_ordered, row3259_values]
def row3271 : SortedNatTree.Tree := (.node 13921 (.node 7621 (.node 5653 (.node 4243 (.node 3361 (.node 3319 .empty .empty) .empty) (.node 5227 .empty .empty)) (.node 6661 (.node 5923 .empty .empty) (.node 7603 .empty .empty))) (.node 9343 (.node 9013 (.node 8377 .empty .empty) (.node 9199 .empty .empty)) (.node 10711 (.node 9697 .empty .empty) (.node 12763 .empty .empty)))) (.node 19927 (.node 17107 (.node 16417 (.node 16249 (.node 14503 .empty .empty) .empty) (.node 16759 .empty .empty)) (.node 18481 (.node 17911 .empty .empty) (.node 19231 .empty .empty))) (.node 21187 (.node 20389 (.node 20089 .empty .empty) (.node 21169 .empty .empty)) (.node 22441 (.node 22153 .empty .empty) (.node 22573 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3271_ordered : SortedNatTree.ordered row3271 = true := by decide
set_option maxRecDepth 4096 in
theorem row3271_values : SortedNatTree.values row3271 = Data.row3271 := rfl
theorem row3271_checked : SortedNatTree.contains row3271 = Data.row3271.contains := by
  rw [SortedNatTree.contains_eq_list row3271 row3271_ordered, row3271_values]
def row3299 : SortedNatTree.Tree := (.node 12119 (.node 7043 (.node 6011 (.node 4523 (.node 4019 (.node 3767 .empty .empty) .empty) (.node 4817 .empty .empty)) (.node 6869 (.node 6203 .empty .empty) (.node 6959 .empty .empty))) (.node 10247 (.node 8447 (.node 8429 (.node 7499 .empty .empty) .empty) (.node 9203 .empty .empty)) (.node 11927 (.node 11213 .empty .empty) (.node 12041 .empty .empty)))) (.node 16073 (.node 15161 (.node 13187 (.node 13109 (.node 12269 .empty .empty) .empty) (.node 15101 .empty .empty)) (.node 15359 (.node 15227 .empty .empty) (.node 15581 .empty .empty))) (.node 19727 (.node 17807 (.node 17417 .empty .empty) (.node 19013 .empty .empty)) (.node 20117 (.node 19973 .empty .empty) (.node 22391 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3299_ordered : SortedNatTree.ordered row3299 = true := by decide
set_option maxRecDepth 4096 in
theorem row3299_values : SortedNatTree.values row3299 = Data.row3299 := rfl
theorem row3299_checked : SortedNatTree.contains row3299 = Data.row3299.contains := by
  rw [SortedNatTree.contains_eq_list row3299 row3299_ordered, row3299_values]
def row3301 : SortedNatTree.Tree := (.node 13219 (.node 6961 (.node 4651 (.node 4447 (.node 4363 (.node 3517 .empty .empty) .empty) (.node 4567 .empty .empty)) (.node 5881 (.node 5779 .empty .empty) (.node 6343 .empty .empty))) (.node 10531 (.node 7297 (.node 6967 .empty .empty) (.node 8893 .empty .empty)) (.node 12421 (.node 10861 .empty .empty) (.node 12637 .empty .empty)))) (.node 18583 (.node 16333 (.node 14593 (.node 14011 (.node 13879 .empty .empty) .empty) (.node 15031 .empty .empty)) (.node 17239 (.node 16651 .empty .empty) (.node 18541 .empty .empty))) (.node 21061 (.node 20521 (.node 20143 .empty .empty) (.node 20743 .empty .empty)) (.node 21991 (.node 21193 .empty .empty) (.node 22669 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3301_ordered : SortedNatTree.ordered row3301 = true := by decide
set_option maxRecDepth 4096 in
theorem row3301_values : SortedNatTree.values row3301 = Data.row3301 := rfl
theorem row3301_checked : SortedNatTree.contains row3301 = Data.row3301.contains := by
  rw [SortedNatTree.contains_eq_list row3301 row3301_ordered, row3301_values]
def row3307 : SortedNatTree.Tree := (.node 9973 (.node 6661 (.node 5737 (.node 4519 (.node 3583 (.node 3331 .empty .empty) .empty) (.node 5443 .empty .empty)) (.node 6367 (.node 6151 (.node 5953 .empty .empty) .empty) (.node 6619 .empty .empty))) (.node 9091 (.node 7411 (.node 7321 (.node 7039 .empty .empty) .empty) (.node 8713 .empty .empty)) (.node 9391 (.node 9349 (.node 9181 .empty .empty) .empty) (.node 9643 .empty .empty)))) (.node 15313 (.node 12073 (.node 10651 (.node 10273 (.node 10099 .empty .empty) .empty) (.node 12037 .empty .empty)) (.node 15031 (.node 14431 (.node 14389 .empty .empty) .empty) (.node 15199 .empty .empty))) (.node 21499 (.node 18457 (.node 18301 (.node 18133 .empty .empty) .empty) (.node 20407 .empty .empty)) (.node 21871 (.node 21529 .empty .empty) (.node 22501 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3307_ordered : SortedNatTree.ordered row3307 = true := by decide
set_option maxRecDepth 4096 in
theorem row3307_values : SortedNatTree.values row3307 = Data.row3307 := rfl
theorem row3307_checked : SortedNatTree.contains row3307 = Data.row3307.contains := by
  rw [SortedNatTree.contains_eq_list row3307 row3307_ordered, row3307_values]
def row3313 : SortedNatTree.Tree := (.node 12301 (.node 6421 (.node 4663 (.node 4111 (.node 3733 (.node 3673 .empty .empty) .empty) (.node 4159 .empty .empty)) (.node 4831 (.node 4723 .empty .empty) (.node 4951 .empty .empty))) (.node 8629 (.node 8059 (.node 6871 (.node 6637 .empty .empty) .empty) (.node 8089 .empty .empty)) (.node 9349 (.node 8779 .empty .empty) (.node 9739 .empty .empty)))) (.node 18541 (.node 14923 (.node 13831 (.node 12781 (.node 12601 .empty .empty) .empty) (.node 14449 .empty .empty)) (.node 17299 (.node 15679 .empty .empty) (.node 17683 .empty .empty))) (.node 20551 (.node 19507 (.node 19447 (.node 18919 .empty .empty) .empty) (.node 20173 .empty .empty)) (.node 21121 (.node 21061 .empty .empty) (.node 22447 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3313_ordered : SortedNatTree.ordered row3313 = true := by decide
set_option maxRecDepth 4096 in
theorem row3313_values : SortedNatTree.values row3313 = Data.row3313 := rfl
theorem row3313_checked : SortedNatTree.contains row3313 = Data.row3313.contains := by
  rw [SortedNatTree.contains_eq_list row3313 row3313_ordered, row3313_values]
def row3319 : SortedNatTree.Tree := (.node 11617 (.node 6577 (.node 4507 (.node 4339 (.node 3613 (.node 3373 .empty .empty) .empty) (.node 4363 .empty .empty)) (.node 6367 (.node 5419 (.node 5209 .empty .empty) .empty) (.node 6481 .empty .empty))) (.node 8647 (.node 7951 (.node 7669 (.node 7321 .empty .empty) .empty) (.node 8101 .empty .empty)) (.node 9871 (.node 9391 (.node 9151 .empty .empty) .empty) (.node 10903 .empty .empty)))) (.node 16561 (.node 13399 (.node 12799 (.node 12289 (.node 11731 .empty .empty) .empty) (.node 13297 .empty .empty)) (.node 15217 (.node 14683 (.node 13723 .empty .empty) .empty) (.node 15373 .empty .empty))) (.node 20509 (.node 18973 (.node 17791 (.node 17107 .empty .empty) .empty) (.node 19069 .empty .empty)) (.node 21529 (.node 21157 .empty .empty) (.node 21727 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3319_ordered : SortedNatTree.ordered row3319 = true := by decide
set_option maxRecDepth 4096 in
theorem row3319_values : SortedNatTree.values row3319 = Data.row3319 := rfl
theorem row3319_checked : SortedNatTree.contains row3319 = Data.row3319.contains := by
  rw [SortedNatTree.contains_eq_list row3319 row3319_ordered, row3319_values]
def row3323 : SortedNatTree.Tree := (.node 8861 (.node 4871 (.node 4013 (.node 3803 (.node 3701 (.node 3359 .empty .empty) .empty) (.node 3989 .empty .empty)) (.node 4679 (.node 4583 (.node 4349 .empty .empty) .empty) (.node 4787 .empty .empty))) (.node 7229 (.node 5507 (.node 5393 (.node 5381 .empty .empty) .empty) (.node 6869 .empty .empty)) (.node 7901 (.node 7349 .empty .empty) (.node 8423 .empty .empty)))) (.node 16073 (.node 14561 (.node 12713 (.node 10733 (.node 10037 .empty .empty) .empty) (.node 13151 .empty .empty)) (.node 15923 (.node 15413 .empty .empty) (.node 16067 .empty .empty))) (.node 18731 (.node 17837 (.node 17333 (.node 16253 .empty .empty) .empty) (.node 18587 .empty .empty)) (.node 19913 (.node 19577 .empty .empty) (.node 20411 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3323_ordered : SortedNatTree.ordered row3323 = true := by decide
set_option maxRecDepth 4096 in
theorem row3323_values : SortedNatTree.values row3323 = Data.row3323 := rfl
theorem row3323_checked : SortedNatTree.contains row3323 = Data.row3323.contains := by
  rw [SortedNatTree.contains_eq_list row3323 row3323_ordered, row3323_values]
def row3329 : SortedNatTree.Tree := (.node 11519 (.node 7001 (.node 4967 (.node 4637 (.node 4127 (.node 3929 .empty .empty) .empty) (.node 4649 .empty .empty)) (.node 5441 (.node 5051 .empty .empty) (.node 6359 .empty .empty))) (.node 9887 (.node 8627 (.node 7877 (.node 7109 .empty .empty) .empty) (.node 9437 .empty .empty)) (.node 10559 (.node 10181 .empty .empty) (.node 11351 .empty .empty)))) (.node 16931 (.node 15263 (.node 13967 (.node 12491 (.node 12119 .empty .empty) .empty) (.node 15137 .empty .empty)) (.node 16091 (.node 15629 .empty .empty) (.node 16487 .empty .empty))) (.node 20879 (.node 18587 (.node 18041 .empty .empty) (.node 18743 .empty .empty)) (.node 22229 (.node 21101 .empty .empty) (.node 22307 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3329_ordered : SortedNatTree.ordered row3329 = true := by decide
set_option maxRecDepth 4096 in
theorem row3329_values : SortedNatTree.values row3329 = Data.row3329 := rfl
theorem row3329_checked : SortedNatTree.contains row3329 = Data.row3329.contains := by
  rw [SortedNatTree.contains_eq_list row3329 row3329_ordered, row3329_values]
def row3331 : SortedNatTree.Tree := (.node 9679 (.node 7717 (.node 5197 (.node 4261 (.node 4231 (.node 3559 .empty .empty) .empty) (.node 4903 (.node 4621 .empty .empty) .empty)) (.node 7159 (.node 5701 (.node 5647 .empty .empty) .empty) (.node 7699 .empty .empty))) (.node 8713 (.node 8221 (.node 7993 (.node 7927 .empty .empty) .empty) (.node 8599 .empty .empty)) (.node 9283 (.node 9001 (.node 8737 .empty .empty) .empty) (.node 9421 .empty .empty)))) (.node 16519 (.node 13159 (.node 12613 (.node 12097 (.node 10333 .empty .empty) .empty) (.node 12757 .empty .empty)) (.node 14011 (.node 13933 (.node 13903 .empty .empty) .empty) (.node 14449 .empty .empty))) (.node 18523 (.node 17011 (.node 16747 (.node 16699 .empty .empty) .empty) (.node 18397 .empty .empty)) (.node 20857 (.node 20233 (.node 19009 .empty .empty) .empty) (.node 22483 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3331_ordered : SortedNatTree.ordered row3331 = true := by decide
set_option maxRecDepth 4096 in
theorem row3331_values : SortedNatTree.values row3331 = Data.row3331 := rfl
theorem row3331_checked : SortedNatTree.contains row3331 = Data.row3331.contains := by
  rw [SortedNatTree.contains_eq_list row3331 row3331_ordered, row3331_values]
def row3343 : SortedNatTree.Tree := (.node 14593 (.node 6373 (.node 4561 (.node 3823 (.node 3643 (.node 3391 .empty .empty) .empty) (.node 4483 .empty .empty)) (.node 5449 (.node 4789 (.node 4567 .empty .empty) .empty) (.node 5791 .empty .empty))) (.node 7687 (.node 6883 (.node 6451 (.node 6379 .empty .empty) .empty) (.node 6961 .empty .empty)) (.node 9811 (.node 9397 (.node 8467 .empty .empty) .empty) (.node 13477 .empty .empty)))) (.node 17449 (.node 16273 (.node 15601 (.node 15307 (.node 14767 .empty .empty) .empty) (.node 15901 .empty .empty)) (.node 16903 (.node 16693 (.node 16573 .empty .empty) .empty) (.node 17341 .empty .empty))) (.node 19891 (.node 19477 (.node 17791 (.node 17551 .empty .empty) .empty) (.node 19819 .empty .empty)) (.node 20533 (.node 20407 .empty .empty) (.node 20773 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3343_ordered : SortedNatTree.ordered row3343 = true := by decide
set_option maxRecDepth 4096 in
theorem row3343_values : SortedNatTree.values row3343 = Data.row3343 := rfl
theorem row3343_checked : SortedNatTree.contains row3343 = Data.row3343.contains := by
  rw [SortedNatTree.contains_eq_list row3343 row3343_ordered, row3343_values]
def row3347 : SortedNatTree.Tree := (.node 10427 (.node 6143 (.node 4967 (.node 4349 (.node 4211 (.node 3821 .empty .empty) .empty) (.node 4703 (.node 4673 .empty .empty) .empty)) (.node 6029 (.node 5399 (.node 5333 .empty .empty) .empty) (.node 6101 .empty .empty))) (.node 8231 (.node 7331 (.node 7043 (.node 6959 .empty .empty) .empty) (.node 7793 (.node 7517 .empty .empty) .empty)) (.node 8999 (.node 8693 (.node 8273 .empty .empty) .empty) (.node 10391 .empty .empty)))) (.node 14771 (.node 12821 (.node 12041 (.node 11777 (.node 11519 .empty .empty) .empty) (.node 12671 (.node 12377 .empty .empty) .empty)) (.node 13751 (.node 13619 (.node 13469 .empty .empty) .empty) (.node 14591 .empty .empty))) (.node 19961 (.node 18869 (.node 17987 (.node 15971 .empty .empty) .empty) (.node 19013 .empty .empty)) (.node 21011 (.node 20873 (.node 20411 .empty .empty) .empty) (.node 21377 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3347_ordered : SortedNatTree.ordered row3347 = true := by decide
set_option maxRecDepth 4096 in
theorem row3347_values : SortedNatTree.values row3347 = Data.row3347 := rfl
theorem row3347_checked : SortedNatTree.contains row3347 = Data.row3347.contains := by
  rw [SortedNatTree.contains_eq_list row3347 row3347_ordered, row3347_values]
def row3359 : SortedNatTree.Tree := (.node 11087 (.node 7853 (.node 6599 (.node 5981 (.node 4229 (.node 4217 .empty .empty) .empty) (.node 6257 .empty .empty)) (.node 6779 (.node 6737 .empty .empty) (.node 7649 .empty .empty))) (.node 9521 (.node 8753 (.node 8597 (.node 8447 .empty .empty) .empty) (.node 9281 .empty .empty)) (.node 10247 (.node 9677 .empty .empty) (.node 11003 .empty .empty)))) (.node 18287 (.node 16631 (.node 15683 (.node 12923 (.node 12659 .empty .empty) .empty) (.node 15791 .empty .empty)) (.node 16901 (.node 16871 .empty .empty) (.node 17837 .empty .empty))) (.node 20789 (.node 20177 (.node 19079 .empty .empty) (.node 20297 .empty .empty)) (.node 21143 (.node 20903 .empty .empty) (.node 22637 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3359_ordered : SortedNatTree.ordered row3359 = true := by decide
set_option maxRecDepth 4096 in
theorem row3359_values : SortedNatTree.values row3359 = Data.row3359 := rfl
theorem row3359_checked : SortedNatTree.contains row3359 = Data.row3359.contains := by
  rw [SortedNatTree.contains_eq_list row3359 row3359_ordered, row3359_values]
def row3361 : SortedNatTree.Tree := (.node 11551 (.node 6343 (.node 4951 (.node 3919 (.node 3739 .empty .empty) (.node 4507 .empty .empty)) (.node 5557 (.node 5419 .empty .empty) (.node 6199 .empty .empty))) (.node 8971 (.node 8737 (.node 8719 .empty .empty) (.node 8887 .empty .empty)) (.node 11257 (.node 9811 .empty .empty) .empty))) (.node 17389 (.node 14983 (.node 14083 (.node 12487 .empty .empty) (.node 14533 .empty .empty)) (.node 16069 (.node 15607 .empty .empty) .empty)) (.node 21661 (.node 21277 (.node 20563 .empty .empty) (.node 21613 .empty .empty)) (.node 22543 (.node 22123 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3361_ordered : SortedNatTree.ordered row3361 = true := by decide
set_option maxRecDepth 4096 in
theorem row3361_values : SortedNatTree.values row3361 = Data.row3361 := rfl
theorem row3361_checked : SortedNatTree.contains row3361 = Data.row3361.contains := by
  rw [SortedNatTree.contains_eq_list row3361 row3361_ordered, row3361_values]
def row3371 : SortedNatTree.Tree := (.node 15137 (.node 6029 (.node 4973 (.node 4649 (.node 4229 (.node 3413 .empty .empty) .empty) (.node 4799 .empty .empty)) (.node 5393 (.node 5153 (.node 5009 .empty .empty) .empty) (.node 5849 .empty .empty))) (.node 11273 (.node 9677 (.node 6983 (.node 6359 .empty .empty) .empty) (.node 10259 .empty .empty)) (.node 12101 (.node 12071 (.node 11987 .empty .empty) .empty) (.node 12239 .empty .empty)))) (.node 18443 (.node 16823 (.node 16067 (.node 15809 (.node 15299 .empty .empty) .empty) (.node 16691 .empty .empty)) (.node 17609 (.node 17321 (.node 16931 .empty .empty) .empty) (.node 17747 .empty .empty))) (.node 20549 (.node 19583 (.node 19301 (.node 18461 .empty .empty) .empty) (.node 20507 .empty .empty)) (.node 21383 (.node 20963 .empty .empty) (.node 21737 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3371_ordered : SortedNatTree.ordered row3371 = true := by decide
set_option maxRecDepth 4096 in
theorem row3371_values : SortedNatTree.values row3371 = Data.row3371 := rfl
theorem row3371_checked : SortedNatTree.contains row3371 = Data.row3371.contains := by
  rw [SortedNatTree.contains_eq_list row3371 row3371_ordered, row3371_values]
def row3373 : SortedNatTree.Tree := (.node 12211 (.node 6793 (.node 5557 (.node 4621 (.node 4339 (.node 3631 .empty .empty) .empty) (.node 4957 .empty .empty)) (.node 6067 (.node 5881 .empty .empty) (.node 6247 .empty .empty))) (.node 8923 (.node 8161 (.node 7621 (.node 6907 .empty .empty) .empty) (.node 8233 .empty .empty)) (.node 10657 (.node 9103 .empty .empty) (.node 11287 .empty .empty)))) (.node 18313 (.node 13681 (.node 12739 (.node 12697 (.node 12583 .empty .empty) .empty) (.node 13381 .empty .empty)) (.node 14083 (.node 13807 .empty .empty) (.node 16069 .empty .empty))) (.node 19471 (.node 18493 (.node 18481 .empty .empty) (.node 19009 .empty .empty)) (.node 20023 (.node 19603 .empty .empty) (.node 21727 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3373_ordered : SortedNatTree.ordered row3373 = true := by decide
set_option maxRecDepth 4096 in
theorem row3373_values : SortedNatTree.values row3373 = Data.row3373 := rfl
theorem row3373_checked : SortedNatTree.contains row3373 = Data.row3373.contains := by
  rw [SortedNatTree.contains_eq_list row3373 row3373_ordered, row3373_values]
def row3389 : SortedNatTree.Tree := (.node 13313 (.node 7547 (.node 5153 (.node 4409 (.node 4139 (.node 3803 .empty .empty) .empty) (.node 4799 .empty .empty)) (.node 5879 (.node 5387 .empty .empty) (.node 6449 .empty .empty))) (.node 11369 (.node 7937 (.node 7841 (.node 7727 .empty .empty) .empty) (.node 10601 .empty .empty)) (.node 12251 (.node 11789 .empty .empty) (.node 12653 .empty .empty)))) (.node 17609 (.node 16349 (.node 15173 (.node 14549 (.node 13931 .empty .empty) .empty) (.node 15263 .empty .empty)) (.node 16943 (.node 16901 .empty .empty) (.node 17123 .empty .empty))) (.node 19709 (.node 18521 (.node 18329 .empty .empty) (.node 18803 .empty .empty)) (.node 20759 (.node 20051 .empty .empty) (.node 21773 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3389_ordered : SortedNatTree.ordered row3389 = true := by decide
set_option maxRecDepth 4096 in
theorem row3389_values : SortedNatTree.values row3389 = Data.row3389 := rfl
theorem row3389_checked : SortedNatTree.contains row3389 = Data.row3389.contains := by
  rw [SortedNatTree.contains_eq_list row3389 row3389_ordered, row3389_values]
def row3391 : SortedNatTree.Tree := (.node 11491 (.node 6733 (.node 5227 (.node 3643 (.node 3511 (.node 3433 .empty .empty) .empty) (.node 5113 (.node 5107 .empty .empty) .empty)) (.node 6199 (.node 6079 (.node 5443 .empty .empty) .empty) (.node 6607 (.node 6361 .empty .empty) .empty))) (.node 9439 (.node 7237 (.node 6997 (.node 6907 .empty .empty) .empty) (.node 8431 (.node 8329 .empty .empty) .empty)) (.node 10267 (.node 9817 (.node 9463 .empty .empty) .empty) (.node 10939 (.node 10531 .empty .empty) .empty)))) (.node 15817 (.node 14437 (.node 12547 (.node 12421 (.node 12049 .empty .empty) .empty) (.node 13627 (.node 13597 .empty .empty) .empty)) (.node 14947 (.node 14887 (.node 14767 .empty .empty) .empty) (.node 15739 (.node 15559 .empty .empty) .empty))) (.node 18427 (.node 17209 (.node 16903 (.node 16231 .empty .empty) .empty) (.node 18133 (.node 17341 .empty .empty) .empty)) (.node 21067 (.node 20323 (.node 19861 .empty .empty) .empty) (.node 22369 (.node 21961 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row3391_ordered : SortedNatTree.ordered row3391 = true := by decide
set_option maxRecDepth 4096 in
theorem row3391_values : SortedNatTree.values row3391 = Data.row3391 := rfl
theorem row3391_checked : SortedNatTree.contains row3391 = Data.row3391.contains := by
  rw [SortedNatTree.contains_eq_list row3391 row3391_ordered, row3391_values]
def row3407 : SortedNatTree.Tree := (.node 11621 (.node 8699 (.node 6599 (.node 4889 (.node 4751 (.node 3671 .empty .empty) .empty) (.node 6389 (.node 6299 .empty .empty) .empty)) (.node 7919 (.node 7691 (.node 7079 .empty .empty) .empty) (.node 8669 .empty .empty))) (.node 9743 (.node 9239 (.node 9029 (.node 8819 .empty .empty) .empty) (.node 9467 .empty .empty)) (.node 11411 (.node 10529 (.node 10151 .empty .empty) .empty) (.node 11471 .empty .empty)))) (.node 16427 (.node 14657 (.node 13523 (.node 12689 (.node 12101 .empty .empty) .empty) (.node 13691 .empty .empty)) (.node 15497 (.node 15329 (.node 14747 .empty .empty) .empty) (.node 15527 .empty .empty))) (.node 20147 (.node 19259 (.node 17921 (.node 16619 .empty .empty) .empty) (.node 19433 .empty .empty)) (.node 20981 (.node 20879 (.node 20399 .empty .empty) .empty) (.node 21503 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3407_ordered : SortedNatTree.ordered row3407 = true := by decide
set_option maxRecDepth 4096 in
theorem row3407_values : SortedNatTree.values row3407 = Data.row3407 := rfl
theorem row3407_checked : SortedNatTree.contains row3407 = Data.row3407.contains := by
  rw [SortedNatTree.contains_eq_list row3407 row3407_ordered, row3407_values]
def row3413 : SortedNatTree.Tree := (.node 10559 (.node 6689 (.node 5939 (.node 4649 (.node 3929 (.node 3761 .empty .empty) .empty) (.node 5009 .empty .empty)) (.node 6449 (.node 6101 .empty .empty) (.node 6653 .empty .empty))) (.node 7949 (.node 7043 (.node 7001 .empty .empty) (.node 7211 .empty .empty)) (.node 9173 (.node 8513 .empty .empty) (.node 9461 .empty .empty)))) (.node 15731 (.node 12401 (.node 12239 (.node 11483 (.node 11027 .empty .empty) .empty) (.node 12269 .empty .empty)) (.node 13781 (.node 13331 .empty .empty) (.node 14387 .empty .empty))) (.node 17351 (.node 16007 (.node 15773 .empty .empty) (.node 16193 .empty .empty)) (.node 18077 (.node 17939 .empty .empty) (.node 19997 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3413_ordered : SortedNatTree.ordered row3413 = true := by decide
set_option maxRecDepth 4096 in
theorem row3413_values : SortedNatTree.values row3413 = Data.row3413 := rfl
theorem row3413_checked : SortedNatTree.contains row3413 = Data.row3413.contains := by
  rw [SortedNatTree.contains_eq_list row3413 row3413_ordered, row3413_values]
def row3433 : SortedNatTree.Tree := (.node 11443 (.node 6733 (.node 5233 (.node 4129 (.node 4051 (.node 3643 .empty .empty) .empty) (.node 4987 .empty .empty)) (.node 6451 (.node 6373 (.node 5407 .empty .empty) .empty) (.node 6607 .empty .empty))) (.node 8221 (.node 7741 (.node 7207 (.node 7129 .empty .empty) .empty) (.node 8167 .empty .empty)) (.node 10987 (.node 8539 (.node 8461 .empty .empty) .empty) (.node 11161 .empty .empty)))) (.node 18427 (.node 17041 (.node 15073 (.node 14929 (.node 14887 .empty .empty) .empty) (.node 16063 .empty .empty)) (.node 17551 (.node 17509 (.node 17383 .empty .empty) .empty) (.node 17971 .empty .empty))) (.node 20641 (.node 19927 (.node 19687 (.node 19423 .empty .empty) .empty) (.node 20431 .empty .empty)) (.node 21247 (.node 20983 .empty .empty) (.node 22171 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3433_ordered : SortedNatTree.ordered row3433 = true := by decide
set_option maxRecDepth 4096 in
theorem row3433_values : SortedNatTree.values row3433 = Data.row3433 := rfl
theorem row3433_checked : SortedNatTree.contains row3433 = Data.row3433.contains := by
  rw [SortedNatTree.contains_eq_list row3433 row3433_ordered, row3433_values]
def row3449 : SortedNatTree.Tree := (.node 13649 (.node 7817 (.node 5693 (.node 4451 (.node 4013 .empty .empty) (.node 5501 .empty .empty)) (.node 7433 (.node 6359 .empty .empty) (.node 7457 .empty .empty))) (.node 10193 (.node 8861 (.node 8111 .empty .empty) (.node 8969 .empty .empty)) (.node 11117 (.node 10799 .empty .empty) (.node 13229 .empty .empty)))) (.node 18443 (.node 15359 (.node 14639 (.node 14369 .empty .empty) (.node 14699 .empty .empty)) (.node 16493 (.node 16361 .empty .empty) (.node 16787 .empty .empty))) (.node 20963 (.node 19403 (.node 18713 .empty .empty) (.node 20063 .empty .empty)) (.node 22157 (.node 21107 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3449_ordered : SortedNatTree.ordered row3449 = true := by decide
set_option maxRecDepth 4096 in
theorem row3449_values : SortedNatTree.values row3449 = Data.row3449 := rfl
theorem row3449_checked : SortedNatTree.contains row3449 = Data.row3449.contains := by
  rw [SortedNatTree.contains_eq_list row3449 row3449_ordered, row3449_values]
def row3457 : SortedNatTree.Tree := (.node 10531 (.node 5449 (.node 4759 (.node 4327 (.node 4297 (.node 3499 .empty .empty) .empty) (.node 4567 .empty .empty)) (.node 5197 (.node 5113 (.node 4789 .empty .empty) .empty) (.node 5419 .empty .empty))) (.node 7603 (.node 6679 (.node 6247 (.node 5869 .empty .empty) .empty) (.node 6823 .empty .empty)) (.node 8839 (.node 8377 (.node 8209 .empty .empty) .empty) (.node 9931 .empty .empty)))) (.node 16411 (.node 13033 (.node 11617 (.node 11257 (.node 11071 .empty .empty) .empty) (.node 12109 .empty .empty)) (.node 15907 (.node 15607 (.node 14827 .empty .empty) .empty) (.node 16183 .empty .empty))) (.node 20983 (.node 18481 (.node 18127 (.node 16927 .empty .empty) .empty) (.node 19813 .empty .empty)) (.node 21661 (.node 21157 .empty .empty) (.node 22567 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3457_ordered : SortedNatTree.ordered row3457 = true := by decide
set_option maxRecDepth 4096 in
theorem row3457_values : SortedNatTree.values row3457 = Data.row3457 := rfl
theorem row3457_checked : SortedNatTree.contains row3457 = Data.row3457.contains := by
  rw [SortedNatTree.contains_eq_list row3457 row3457_ordered, row3457_values]
def row3461 : SortedNatTree.Tree := (.node 13037 (.node 7823 (.node 6011 (.node 4967 (.node 4583 (.node 3659 .empty .empty) .empty) (.node 5333 .empty .empty)) (.node 6737 (.node 6221 .empty .empty) (.node 7523 .empty .empty))) (.node 11069 (.node 8543 (.node 8117 .empty .empty) (.node 10559 .empty .empty)) (.node 11813 (.node 11261 .empty .empty) (.node 12953 .empty .empty)))) (.node 17789 (.node 14423 (.node 13877 (.node 13121 .empty .empty) (.node 13967 .empty .empty)) (.node 15269 (.node 15083 .empty .empty) (.node 16091 .empty .empty))) (.node 19373 (.node 18371 (.node 17903 .empty .empty) (.node 18917 .empty .empty)) (.node 21401 (.node 19919 .empty .empty) (.node 21767 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3461_ordered : SortedNatTree.ordered row3461 = true := by decide
set_option maxRecDepth 4096 in
theorem row3461_values : SortedNatTree.values row3461 = Data.row3461 := rfl
theorem row3461_checked : SortedNatTree.contains row3461 = Data.row3461.contains := by
  rw [SortedNatTree.contains_eq_list row3461 row3461_ordered, row3461_values]
def row3463 : SortedNatTree.Tree := (.node 10687 (.node 7411 (.node 5869 (.node 4801 (.node 4447 (.node 4051 .empty .empty) .empty) (.node 5479 (.node 4969 .empty .empty) .empty)) (.node 6211 (.node 6079 (.node 6037 .empty .empty) .empty) (.node 6883 (.node 6337 .empty .empty) .empty))) (.node 9049 (.node 8059 (.node 8053 (.node 8011 .empty .empty) .empty) (.node 9043 (.node 8629 .empty .empty) .empty)) (.node 10099 (.node 9883 (.node 9697 .empty .empty) .empty) (.node 10267 .empty .empty)))) (.node 17599 (.node 15913 (.node 14503 (.node 13567 (.node 11443 .empty .empty) .empty) (.node 15583 (.node 14713 .empty .empty) .empty)) (.node 17053 (.node 17029 (.node 16561 .empty .empty) .empty) (.node 17467 .empty .empty))) (.node 20389 (.node 18919 (.node 17977 (.node 17683 .empty .empty) .empty) (.node 19069 (.node 18979 .empty .empty) .empty)) (.node 21169 (.node 21061 (.node 20743 .empty .empty) .empty) (.node 21601 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3463_ordered : SortedNatTree.ordered row3463 = true := by decide
set_option maxRecDepth 4096 in
theorem row3463_values : SortedNatTree.values row3463 = Data.row3463 := rfl
theorem row3463_checked : SortedNatTree.contains row3463 = Data.row3463.contains := by
  rw [SortedNatTree.contains_eq_list row3463 row3463_ordered, row3463_values]
def row3467 : SortedNatTree.Tree := (.node 7949 (.node 5297 (.node 4817 (.node 4457 (.node 3851 (.node 3623 .empty .empty) .empty) (.node 4799 .empty .empty)) (.node 5231 (.node 5009 (.node 4919 .empty .empty) .empty) (.node 5273 .empty .empty))) (.node 6599 (.node 5807 (.node 5591 (.node 5471 .empty .empty) .empty) (.node 6113 .empty .empty)) (.node 7001 (.node 6959 .empty .empty) (.node 7481 .empty .empty)))) (.node 16631 (.node 9059 (.node 8663 (.node 8573 (.node 8117 .empty .empty) .empty) (.node 8951 .empty .empty)) (.node 10589 (.node 9833 (.node 9479 .empty .empty) .empty) (.node 10691 .empty .empty))) (.node 18947 (.node 18149 (.node 17123 (.node 16661 .empty .empty) .empty) (.node 18311 .empty .empty)) (.node 20123 (.node 19073 .empty .empty) (.node 21911 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3467_ordered : SortedNatTree.ordered row3467 = true := by decide
set_option maxRecDepth 4096 in
theorem row3467_values : SortedNatTree.values row3467 = Data.row3467 := rfl
theorem row3467_checked : SortedNatTree.contains row3467 = Data.row3467.contains := by
  rw [SortedNatTree.contains_eq_list row3467 row3467_ordered, row3467_values]
def row3469 : SortedNatTree.Tree := (.node 11317 (.node 7309 (.node 5503 (.node 4357 (.node 4021 (.node 3517 .empty .empty) .empty) (.node 4903 .empty .empty)) (.node 6703 (.node 5839 .empty .empty) (.node 7219 .empty .empty))) (.node 9403 (.node 8311 (.node 7687 .empty .empty) (.node 8821 .empty .empty)) (.node 10273 (.node 9619 .empty .empty) (.node 10453 .empty .empty)))) (.node 15937 (.node 13513 (.node 12301 (.node 11959 (.node 11839 .empty .empty) .empty) (.node 12799 .empty .empty)) (.node 14107 (.node 14011 .empty .empty) (.node 15643 .empty .empty))) (.node 18757 (.node 16963 (.node 16831 .empty .empty) (.node 18043 .empty .empty)) (.node 19861 (.node 19273 .empty .empty) (.node 21739 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3469_ordered : SortedNatTree.ordered row3469 = true := by decide
set_option maxRecDepth 4096 in
theorem row3469_values : SortedNatTree.values row3469 = Data.row3469 := rfl
theorem row3469_checked : SortedNatTree.contains row3469 = Data.row3469.contains := by
  rw [SortedNatTree.contains_eq_list row3469 row3469_ordered, row3469_values]
def row3491 : SortedNatTree.Tree := (.node 13781 (.node 6221 (.node 4931 (.node 4259 (.node 3947 (.node 3761 .empty .empty) .empty) (.node 4349 .empty .empty)) (.node 5171 (.node 5003 .empty .empty) (.node 5843 .empty .empty))) (.node 8123 (.node 7331 (.node 7121 (.node 6701 .empty .empty) .empty) (.node 8111 .empty .empty)) (.node 10631 (.node 9281 .empty .empty) (.node 11069 .empty .empty)))) (.node 17471 (.node 16319 (.node 15959 (.node 15161 (.node 14009 .empty .empty) .empty) (.node 16073 .empty .empty)) (.node 17159 (.node 17033 .empty .empty) (.node 17333 .empty .empty))) (.node 19559 (.node 18593 (.node 18257 (.node 18089 .empty .empty) .empty) (.node 18899 .empty .empty)) (.node 21317 (.node 19949 .empty .empty) (.node 21863 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3491_ordered : SortedNatTree.ordered row3491 = true := by decide
set_option maxRecDepth 4096 in
theorem row3491_values : SortedNatTree.values row3491 = Data.row3491 := rfl
theorem row3491_checked : SortedNatTree.contains row3491 = Data.row3491.contains := by
  rw [SortedNatTree.contains_eq_list row3491 row3491_ordered, row3491_values]
def row3499 : SortedNatTree.Tree := (.node 12373 (.node 7993 (.node 5449 (.node 5011 (.node 4651 (.node 3613 .empty .empty) .empty) (.node 5407 (.node 5227 .empty .empty) .empty)) (.node 6361 (.node 6277 (.node 5881 .empty .empty) .empty) (.node 7759 (.node 7321 .empty .empty) .empty))) (.node 9619 (.node 8599 (.node 8377 (.node 8191 .empty .empty) .empty) (.node 9343 (.node 9157 .empty .empty) .empty)) (.node 10333 (.node 9931 (.node 9871 .empty .empty) .empty) (.node 11437 .empty .empty)))) (.node 15739 (.node 14437 (.node 13339 (.node 12757 (.node 12487 .empty .empty) .empty) (.node 14419 (.node 14293 .empty .empty) .empty)) (.node 15217 (.node 14797 (.node 14713 .empty .empty) .empty) (.node 15331 (.node 15277 .empty .empty) .empty))) (.node 19387 (.node 18199 (.node 17977 (.node 16381 .empty .empty) .empty) (.node 19081 (.node 18691 .empty .empty) .empty)) (.node 20521 (.node 20101 (.node 20047 .empty .empty) .empty) (.node 22453 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3499_ordered : SortedNatTree.ordered row3499 = true := by decide
set_option maxRecDepth 4096 in
theorem row3499_values : SortedNatTree.values row3499 = Data.row3499 := rfl
theorem row3499_checked : SortedNatTree.contains row3499 = Data.row3499.contains := by
  rw [SortedNatTree.contains_eq_list row3499 row3499_ordered, row3499_values]
def row3511 : SortedNatTree.Tree := (.node 12451 (.node 8527 (.node 7561 (.node 5233 (.node 4153 (.node 3889 .empty .empty) .empty) (.node 6967 .empty .empty)) (.node 8011 (.node 7963 (.node 7879 .empty .empty) .empty) (.node 8089 .empty .empty))) (.node 11083 (.node 9439 (.node 9283 (.node 9181 .empty .empty) .empty) (.node 11071 .empty .empty)) (.node 11161 (.node 11113 .empty .empty) (.node 12007 .empty .empty)))) (.node 18433 (.node 14563 (.node 13591 (.node 13417 (.node 12919 .empty .empty) .empty) (.node 14173 .empty .empty)) (.node 16693 (.node 16231 (.node 15601 .empty .empty) .empty) (.node 17707 .empty .empty))) (.node 20233 (.node 19429 (.node 19009 (.node 18637 .empty .empty) .empty) (.node 19861 .empty .empty)) (.node 20899 (.node 20857 .empty .empty) (.node 21997 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3511_ordered : SortedNatTree.ordered row3511 = true := by decide
set_option maxRecDepth 4096 in
theorem row3511_values : SortedNatTree.values row3511 = Data.row3511 := rfl
theorem row3511_checked : SortedNatTree.contains row3511 = Data.row3511.contains := by
  rw [SortedNatTree.contains_eq_list row3511 row3511_ordered, row3511_values]
def row3517 : SortedNatTree.Tree := (.node 10663 (.node 7669 (.node 6907 (.node 5659 (.node 4177 (.node 4099 .empty .empty) .empty) (.node 6829 .empty .empty)) (.node 7507 (.node 7159 .empty .empty) (.node 7621 .empty .empty))) (.node 8599 (.node 8161 (.node 7717 .empty .empty) (.node 8263 .empty .empty)) (.node 9769 (.node 9433 .empty .empty) (.node 9883 .empty .empty)))) (.node 19813 (.node 14401 (.node 13597 (.node 13177 (.node 11149 .empty .empty) .empty) (.node 13759 .empty .empty)) (.node 15349 (.node 15319 .empty .empty) (.node 15817 .empty .empty))) (.node 21499 (.node 21397 (.node 20731 .empty .empty) (.node 21481 .empty .empty)) (.node 21817 (.node 21529 .empty .empty) (.node 22039 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3517_ordered : SortedNatTree.ordered row3517 = true := by decide
set_option maxRecDepth 4096 in
theorem row3517_values : SortedNatTree.values row3517 = Data.row3517 := rfl
theorem row3517_checked : SortedNatTree.contains row3517 = Data.row3517.contains := by
  rw [SortedNatTree.contains_eq_list row3517 row3517_ordered, row3517_values]
def row3527 : SortedNatTree.Tree := (.node 12911 (.node 8669 (.node 6803 (.node 4649 (.node 4457 (.node 3533 .empty .empty) .empty) (.node 6551 .empty .empty)) (.node 7901 (.node 7013 .empty .empty) (.node 8429 .empty .empty))) (.node 10979 (.node 9239 (.node 8861 .empty .empty) (.node 9323 .empty .empty)) (.node 11519 (.node 11351 .empty .empty) (.node 12377 .empty .empty)))) (.node 19571 (.node 17471 (.node 16421 (.node 14897 .empty .empty) (.node 16631 .empty .empty)) (.node 18617 (.node 17837 .empty .empty) (.node 19163 .empty .empty))) (.node 20747 (.node 20327 (.node 19709 .empty .empty) (.node 20507 .empty .empty)) (.node 21587 (.node 21059 .empty .empty) (.node 22229 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3527_ordered : SortedNatTree.ordered row3527 = true := by decide
set_option maxRecDepth 4096 in
theorem row3527_values : SortedNatTree.values row3527 = Data.row3527 := rfl
theorem row3527_checked : SortedNatTree.contains row3527 = Data.row3527.contains := by
  rw [SortedNatTree.contains_eq_list row3527 row3527_ordered, row3527_values]
def row3529 : SortedNatTree.Tree := (.node 11173 (.node 6553 (.node 5527 (.node 4657 (.node 4423 (.node 3943 .empty .empty) .empty) (.node 5011 (.node 4969 .empty .empty) .empty)) (.node 6397 (.node 5851 (.node 5569 .empty .empty) .empty) (.node 6427 .empty .empty))) (.node 8707 (.node 8059 (.node 7393 (.node 6733 .empty .empty) .empty) (.node 8521 .empty .empty)) (.node 10069 (.node 9619 (.node 9403 .empty .empty) .empty) (.node 11059 .empty .empty)))) (.node 16981 (.node 13567 (.node 12211 (.node 11437 (.node 11257 .empty .empty) .empty) (.node 13339 .empty .empty)) (.node 16633 (.node 15217 (.node 14143 .empty .empty) .empty) (.node 16699 .empty .empty))) (.node 20983 (.node 20719 (.node 20521 (.node 19867 .empty .empty) .empty) (.node 20857 .empty .empty)) (.node 21589 (.node 21433 (.node 21013 .empty .empty) .empty) (.node 22303 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3529_ordered : SortedNatTree.ordered row3529 = true := by decide
set_option maxRecDepth 4096 in
theorem row3529_values : SortedNatTree.values row3529 = Data.row3529 := rfl
theorem row3529_checked : SortedNatTree.contains row3529 = Data.row3529.contains := by
  rw [SortedNatTree.contains_eq_list row3529 row3529_ordered, row3529_values]
def row3533 : SortedNatTree.Tree := (.node 10163 (.node 7079 (.node 4703 (.node 4583 (.node 4133 (.node 3593 .empty .empty) .empty) (.node 4679 .empty .empty)) (.node 6113 (.node 5483 (.node 5099 .empty .empty) .empty) (.node 7019 .empty .empty))) (.node 9173 (.node 8861 (.node 8273 (.node 7607 .empty .empty) .empty) (.node 8951 .empty .empty)) (.node 9341 (.node 9323 .empty .empty) (.node 9539 .empty .empty)))) (.node 15227 (.node 10709 (.node 10313 (.node 10289 (.node 10169 .empty .empty) .empty) (.node 10691 .empty .empty)) (.node 13613 (.node 11171 (.node 11087 .empty .empty) .empty) (.node 13967 .empty .empty))) (.node 20483 (.node 19937 (.node 18947 (.node 18839 .empty .empty) .empty) (.node 20399 .empty .empty)) (.node 22247 (.node 20771 .empty .empty) (.node 22349 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3533_ordered : SortedNatTree.ordered row3533 = true := by decide
set_option maxRecDepth 4096 in
theorem row3533_values : SortedNatTree.values row3533 = Data.row3533 := rfl
theorem row3533_checked : SortedNatTree.contains row3533 = Data.row3533.contains := by
  rw [SortedNatTree.contains_eq_list row3533 row3533_ordered, row3533_values]
def row3539 : SortedNatTree.Tree := (.node 13997 (.node 7829 (.node 5471 (.node 4649 (.node 4013 (.node 3671 .empty .empty) .empty) (.node 5441 .empty .empty)) (.node 6521 (.node 5861 .empty .empty) (.node 7001 .empty .empty))) (.node 11981 (.node 9803 (.node 9467 (.node 9137 .empty .empty) .empty) (.node 10181 .empty .empty)) (.node 12941 (.node 12899 .empty .empty) (.node 13331 .empty .empty)))) (.node 18077 (.node 16889 (.node 16007 (.node 15383 (.node 15083 .empty .empty) .empty) (.node 16103 .empty .empty)) (.node 17903 (.node 17021 .empty .empty) (.node 17909 .empty .empty))) (.node 20627 (.node 19457 (.node 18839 (.node 18119 .empty .empty) .empty) (.node 19751 .empty .empty)) (.node 21767 (.node 21023 .empty .empty) (.node 22193 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3539_ordered : SortedNatTree.ordered row3539 = true := by decide
set_option maxRecDepth 4096 in
theorem row3539_values : SortedNatTree.values row3539 = Data.row3539 := rfl
theorem row3539_checked : SortedNatTree.contains row3539 = Data.row3539.contains := by
  rw [SortedNatTree.contains_eq_list row3539 row3539_ordered, row3539_values]
def row3541 : SortedNatTree.Tree := (.node 12757 (.node 9001 (.node 7489 (.node 7243 (.node 4567 .empty .empty) (.node 7351 .empty .empty)) (.node 8011 (.node 7951 .empty .empty) (.node 8263 .empty .empty))) (.node 10567 (.node 10177 (.node 9187 .empty .empty) (.node 10321 .empty .empty)) (.node 10651 (.node 10597 .empty .empty) (.node 11701 .empty .empty)))) (.node 18691 (.node 13789 (.node 13063 (.node 12919 .empty .empty) (.node 13537 .empty .empty)) (.node 17209 (.node 14557 .empty .empty) (.node 18169 .empty .empty))) (.node 20563 (.node 20023 (.node 19687 .empty .empty) (.node 20089 .empty .empty)) (.node 21787 (.node 20749 .empty .empty) (.node 22123 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3541_ordered : SortedNatTree.ordered row3541 = true := by decide
set_option maxRecDepth 4096 in
theorem row3541_values : SortedNatTree.values row3541 = Data.row3541 := rfl
theorem row3541_checked : SortedNatTree.contains row3541 = Data.row3541.contains := by
  rw [SortedNatTree.contains_eq_list row3541 row3541_ordered, row3541_values]
def row3547 : SortedNatTree.Tree := (.node 11839 (.node 7243 (.node 6151 (.node 5449 (.node 4357 (.node 3643 .empty .empty) .empty) (.node 5827 .empty .empty)) (.node 6709 (.node 6481 .empty .empty) (.node 7213 .empty .empty))) (.node 9319 (.node 8461 (.node 8431 (.node 7993 .empty .empty) .empty) (.node 9043 .empty .empty)) (.node 9817 (.node 9733 .empty .empty) (.node 10333 .empty .empty)))) (.node 18061 (.node 13537 (.node 12703 (.node 12637 (.node 12343 .empty .empty) .empty) (.node 12907 .empty .empty)) (.node 15937 (.node 15289 .empty .empty) (.node 16633 .empty .empty))) (.node 18979 (.node 18451 (.node 18433 .empty .empty) (.node 18457 .empty .empty)) (.node 22129 (.node 19423 .empty .empty) (.node 22483 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3547_ordered : SortedNatTree.ordered row3547 = true := by decide
set_option maxRecDepth 4096 in
theorem row3547_values : SortedNatTree.values row3547 = Data.row3547 := rfl
theorem row3547_checked : SortedNatTree.contains row3547 = Data.row3547.contains := by
  rw [SortedNatTree.contains_eq_list row3547 row3547_ordered, row3547_values]
def row3557 : SortedNatTree.Tree := (.node 12491 (.node 6551 (.node 5279 (.node 5021 (.node 4139 (.node 3803 .empty .empty) .empty) (.node 5051 .empty .empty)) (.node 6131 (.node 5783 (.node 5417 .empty .empty) .empty) (.node 6329 .empty .empty))) (.node 10007 (.node 9629 (.node 9437 (.node 8681 .empty .empty) .empty) (.node 9857 .empty .empty)) (.node 11717 (.node 10529 .empty .empty) (.node 11789 .empty .empty)))) (.node 17987 (.node 14897 (.node 13751 (.node 13679 (.node 13037 .empty .empty) .empty) (.node 14321 .empty .empty)) (.node 16883 (.node 16811 .empty .empty) (.node 16937 .empty .empty))) (.node 20063 (.node 19421 (.node 19391 (.node 19163 .empty .empty) .empty) (.node 19751 .empty .empty)) (.node 21503 (.node 21269 .empty .empty) (.node 21701 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3557_ordered : SortedNatTree.ordered row3557 = true := by decide
set_option maxRecDepth 4096 in
theorem row3557_values : SortedNatTree.values row3557 = Data.row3557 := rfl
theorem row3557_checked : SortedNatTree.contains row3557 = Data.row3557.contains := by
  rw [SortedNatTree.contains_eq_list row3557 row3557_ordered, row3557_values]
def row3559 : SortedNatTree.Tree := (.node 10099 (.node 6961 (.node 5449 (.node 3769 (.node 3709 (.node 3583 .empty .empty) .empty) (.node 4021 (.node 3931 .empty .empty) .empty)) (.node 6607 (.node 6037 (.node 5869 .empty .empty) .empty) (.node 6619 .empty .empty))) (.node 9337 (.node 8053 (.node 7417 (.node 7369 .empty .empty) .empty) (.node 8707 .empty .empty)) (.node 9721 (.node 9511 (.node 9397 .empty .empty) .empty) (.node 9973 .empty .empty)))) (.node 16411 (.node 13627 (.node 11161 (.node 10987 (.node 10771 .empty .empty) .empty) (.node 13219 (.node 13177 .empty .empty) .empty)) (.node 15649 (.node 15139 (.node 14341 .empty .empty) .empty) (.node 15727 .empty .empty))) (.node 19441 (.node 18637 (.node 18217 (.node 16657 .empty .empty) .empty) (.node 18679 .empty .empty)) (.node 20743 (.node 19843 (.node 19507 .empty .empty) .empty) (.node 21859 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3559_ordered : SortedNatTree.ordered row3559 = true := by decide
set_option maxRecDepth 4096 in
theorem row3559_values : SortedNatTree.values row3559 = Data.row3559 := rfl
theorem row3559_checked : SortedNatTree.contains row3559 = Data.row3559.contains := by
  rw [SortedNatTree.contains_eq_list row3559 row3559_ordered, row3559_values]
def row3571 : SortedNatTree.Tree := (.node 13327 (.node 7699 (.node 4969 (.node 4801 (.node 4363 (.node 4111 .empty .empty) .empty) (.node 4957 .empty .empty)) (.node 6451 (.node 6007 (.node 5023 .empty .empty) .empty) (.node 6883 .empty .empty))) (.node 11329 (.node 9067 (.node 9001 (.node 7963 .empty .empty) .empty) (.node 9181 .empty .empty)) (.node 13063 (.node 12289 (.node 11959 .empty .empty) .empty) (.node 13147 .empty .empty)))) (.node 19699 (.node 17203 (.node 14503 (.node 14449 (.node 14143 .empty .empty) .empty) (.node 15907 .empty .empty)) (.node 19387 (.node 18127 (.node 17581 .empty .empty) .empty) (.node 19501 .empty .empty))) (.node 21211 (.node 20359 (.node 20107 (.node 19753 .empty .empty) .empty) (.node 20773 .empty .empty)) (.node 22063 (.node 21871 (.node 21517 .empty .empty) .empty) (.node 22159 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3571_ordered : SortedNatTree.ordered row3571 = true := by decide
set_option maxRecDepth 4096 in
theorem row3571_values : SortedNatTree.values row3571 = Data.row3571 := rfl
theorem row3571_checked : SortedNatTree.contains row3571 = Data.row3571.contains := by
  rw [SortedNatTree.contains_eq_list row3571 row3571_ordered, row3571_values]
def row3581 : SortedNatTree.Tree := (.node 11657 (.node 5849 (.node 4643 (.node 4133 (.node 4091 (.node 3851 .empty .empty) .empty) (.node 4421 (.node 4139 .empty .empty) .empty)) (.node 5303 (.node 4931 (.node 4679 .empty .empty) .empty) (.node 5387 .empty .empty))) (.node 7253 (.node 6689 (.node 6101 (.node 5903 .empty .empty) .empty) (.node 6983 (.node 6701 .empty .empty) .empty)) (.node 9803 (.node 8807 (.node 7829 .empty .empty) .empty) (.node 9941 .empty .empty)))) (.node 17351 (.node 14411 (.node 12791 (.node 12401 (.node 12239 .empty .empty) .empty) (.node 13487 (.node 12917 .empty .empty) .empty)) (.node 16487 (.node 16223 (.node 16217 .empty .empty) .empty) (.node 16763 .empty .empty))) (.node 19997 (.node 19469 (.node 19031 (.node 18443 .empty .empty) .empty) (.node 19763 (.node 19661 .empty .empty) .empty)) (.node 21011 (.node 20849 (.node 20753 .empty .empty) .empty) (.node 21143 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3581_ordered : SortedNatTree.ordered row3581 = true := by decide
set_option maxRecDepth 4096 in
theorem row3581_values : SortedNatTree.values row3581 = Data.row3581 := rfl
theorem row3581_checked : SortedNatTree.contains row3581 = Data.row3581.contains := by
  rw [SortedNatTree.contains_eq_list row3581 row3581_ordered, row3581_values]
def row3583 : SortedNatTree.Tree := (.node 12457 (.node 7759 (.node 5857 (.node 4651 (.node 4423 (.node 4177 .empty .empty) .empty) (.node 5689 (.node 5179 .empty .empty) .empty)) (.node 6247 (.node 6067 (.node 5953 .empty .empty) .empty) (.node 7057 .empty .empty))) (.node 9631 (.node 8527 (.node 8191 (.node 7951 .empty .empty) .empty) (.node 8779 .empty .empty)) (.node 10837 (.node 10723 (.node 9787 .empty .empty) .empty) (.node 11887 .empty .empty)))) (.node 15649 (.node 13831 (.node 12979 (.node 12613 (.node 12517 .empty .empty) .empty) (.node 13693 .empty .empty)) (.node 14851 (.node 14713 (.node 13933 .empty .empty) .empty) (.node 15361 .empty .empty))) (.node 19237 (.node 18757 (.node 16741 (.node 16267 .empty .empty) .empty) (.node 19141 .empty .empty)) (.node 22051 (.node 21661 (.node 20731 .empty .empty) .empty) (.node 22147 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3583_ordered : SortedNatTree.ordered row3583 = true := by decide
set_option maxRecDepth 4096 in
theorem row3583_values : SortedNatTree.values row3583 = Data.row3583 := rfl
theorem row3583_checked : SortedNatTree.contains row3583 = Data.row3583.contains := by
  rw [SortedNatTree.contains_eq_list row3583 row3583_ordered, row3583_values]
def row3593 : SortedNatTree.Tree := (.node 11027 (.node 7877 (.node 5279 (.node 4517 (.node 4001 (.node 3719 .empty .empty) .empty) (.node 5087 (.node 5021 .empty .empty) .empty)) (.node 6311 (.node 5903 (.node 5693 .empty .empty) .empty) (.node 7529 .empty .empty))) (.node 8831 (.node 8681 (.node 8297 (.node 7907 .empty .empty) .empty) (.node 8807 .empty .empty)) (.node 10529 (.node 10067 (.node 9473 .empty .empty) .empty) (.node 10607 .empty .empty)))) (.node 16553 (.node 14621 (.node 13469 (.node 13049 (.node 13001 .empty .empty) .empty) (.node 14519 (.node 13679 .empty .empty) .empty)) (.node 15773 (.node 15629 (.node 15083 .empty .empty) .empty) (.node 16007 .empty .empty))) (.node 18461 (.node 17291 (.node 17207 (.node 17099 .empty .empty) .empty) (.node 18131 .empty .empty)) (.node 20249 (.node 20147 (.node 19391 .empty .empty) .empty) (.node 20879 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3593_ordered : SortedNatTree.ordered row3593 = true := by decide
set_option maxRecDepth 4096 in
theorem row3593_values : SortedNatTree.values row3593 = Data.row3593 := rfl
theorem row3593_checked : SortedNatTree.contains row3593 = Data.row3593.contains := by
  rw [SortedNatTree.contains_eq_list row3593 row3593_ordered, row3593_values]
def row3607 : SortedNatTree.Tree := (.node 14737 (.node 7351 (.node 6367 (.node 5023 (.node 4099 .empty .empty) (.node 5077 .empty .empty)) (.node 6571 (.node 6379 .empty .empty) (.node 6577 .empty .empty))) (.node 8521 (.node 8089 (.node 7669 .empty .empty) (.node 8101 .empty .empty)) (.node 8941 (.node 8803 .empty .empty) (.node 13033 .empty .empty)))) (.node 18517 (.node 16063 (.node 15259 (.node 14797 .empty .empty) (.node 15307 .empty .empty)) (.node 17011 (.node 16381 .empty .empty) (.node 18493 .empty .empty))) (.node 20509 (.node 19231 (.node 18859 .empty .empty) (.node 19717 .empty .empty)) (.node 22279 (.node 21379 .empty .empty) (.node 22381 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3607_ordered : SortedNatTree.ordered row3607 = true := by decide
set_option maxRecDepth 4096 in
theorem row3607_values : SortedNatTree.values row3607 = Data.row3607 := rfl
theorem row3607_checked : SortedNatTree.contains row3607 = Data.row3607.contains := by
  rw [SortedNatTree.contains_eq_list row3607 row3607_ordered, row3607_values]
def row3613 : SortedNatTree.Tree := (.node 14029 (.node 8353 (.node 5233 (.node 4831 (.node 3697 .empty .empty) (.node 5023 .empty .empty)) (.node 5683 (.node 5623 .empty .empty) (.node 5821 .empty .empty))) (.node 9613 (.node 8887 (.node 8803 .empty .empty) (.node 9151 .empty .empty)) (.node 13159 (.node 10987 .empty .empty) .empty))) (.node 15919 (.node 14983 (.node 14293 (.node 14071 .empty .empty) (.node 14851 .empty .empty)) (.node 15289 (.node 15061 .empty .empty) (.node 15313 .empty .empty))) (.node 18973 (.node 17791 (.node 17041 .empty .empty) (.node 18229 .empty .empty)) (.node 21739 (.node 20269 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3613_ordered : SortedNatTree.ordered row3613 = true := by decide
set_option maxRecDepth 4096 in
theorem row3613_values : SortedNatTree.values row3613 = Data.row3613 := rfl
theorem row3613_checked : SortedNatTree.contains row3613 = Data.row3613.contains := by
  rw [SortedNatTree.contains_eq_list row3613 row3613_ordered, row3613_values]
def row3617 : SortedNatTree.Tree := (.node 11633 (.node 6089 (.node 4703 (.node 3923 (.node 3803 (.node 3677 .empty .empty) .empty) (.node 4643 (.node 4517 .empty .empty) .empty)) (.node 5651 (.node 5333 (.node 5003 .empty .empty) .empty) (.node 5669 .empty .empty))) (.node 10313 (.node 8231 (.node 7901 (.node 7607 .empty .empty) .empty) (.node 9413 (.node 8291 .empty .empty) .empty)) (.node 11279 (.node 10589 (.node 10499 .empty .empty) .empty) (.node 11579 .empty .empty)))) (.node 18341 (.node 15329 (.node 13043 (.node 12917 (.node 11807 .empty .empty) .empty) (.node 15077 (.node 14657 .empty .empty) .empty)) (.node 16673 (.node 16001 (.node 15461 .empty .empty) .empty) (.node 17747 .empty .empty))) (.node 21467 (.node 20753 (.node 20483 (.node 18443 .empty .empty) .empty) (.node 20771 .empty .empty)) (.node 22259 (.node 22157 (.node 21821 .empty .empty) .empty) (.node 22367 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3617_ordered : SortedNatTree.ordered row3617 = true := by decide
set_option maxRecDepth 4096 in
theorem row3617_values : SortedNatTree.values row3617 = Data.row3617 := rfl
theorem row3617_checked : SortedNatTree.contains row3617 = Data.row3617.contains := by
  rw [SortedNatTree.contains_eq_list row3617 row3617_ordered, row3617_values]
def row3623 : SortedNatTree.Tree := (.node 14369 (.node 6521 (.node 4271 (.node 4091 (.node 4049 .empty .empty) (.node 4127 .empty .empty)) (.node 5981 (.node 4967 .empty .empty) (.node 6449 .empty .empty))) (.node 9173 (.node 8963 (.node 8123 .empty .empty) (.node 9059 .empty .empty)) (.node 11867 (.node 9719 .empty .empty) (.node 12227 .empty .empty)))) (.node 18947 (.node 17393 (.node 16193 (.node 15263 .empty .empty) (.node 16901 .empty .empty)) (.node 18401 (.node 18251 .empty .empty) (.node 18413 .empty .empty))) (.node 21107 (.node 19973 (.node 19469 .empty .empty) (.node 20921 .empty .empty)) (.node 21821 (.node 21467 .empty .empty) (.node 21977 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3623_ordered : SortedNatTree.ordered row3623 = true := by decide
set_option maxRecDepth 4096 in
theorem row3623_values : SortedNatTree.values row3623 = Data.row3623 := rfl
theorem row3623_checked : SortedNatTree.contains row3623 = Data.row3623.contains := by
  rw [SortedNatTree.contains_eq_list row3623 row3623_ordered, row3623_values]
def row3631 : SortedNatTree.Tree := (.node 14083 (.node 8377 (.node 6691 (.node 5347 (.node 5119 (.node 4513 .empty .empty) .empty) (.node 6547 .empty .empty)) (.node 7789 (.node 7507 .empty .empty) (.node 8017 .empty .empty))) (.node 12577 (.node 8923 (.node 8839 .empty .empty) (.node 11383 .empty .empty)) (.node 13537 (.node 13411 .empty .empty) (.node 13687 .empty .empty)))) (.node 18439 (.node 16987 (.node 16063 (.node 14197 (.node 14107 .empty .empty) .empty) (.node 16519 .empty .empty)) (.node 17659 (.node 17389 .empty .empty) (.node 18397 .empty .empty))) (.node 21397 (.node 19927 (.node 19417 .empty .empty) (.node 21019 .empty .empty)) (.node 21943 (.node 21937 .empty .empty) (.node 22093 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3631_ordered : SortedNatTree.ordered row3631 = true := by decide
set_option maxRecDepth 4096 in
theorem row3631_values : SortedNatTree.values row3631 = Data.row3631 := rfl
theorem row3631_checked : SortedNatTree.contains row3631 = Data.row3631.contains := by
  rw [SortedNatTree.contains_eq_list row3631 row3631_ordered, row3631_values]
def row3637 : SortedNatTree.Tree := (.node 11503 (.node 6871 (.node 5659 (.node 5407 (.node 3643 .empty .empty) (.node 5503 .empty .empty)) (.node 6343 (.node 5839 .empty .empty) (.node 6841 .empty .empty))) (.node 8803 (.node 8059 (.node 7639 .empty .empty) (.node 8677 .empty .empty)) (.node 9697 (.node 9649 .empty .empty) (.node 10771 .empty .empty)))) (.node 15139 (.node 13003 (.node 11863 (.node 11827 .empty .empty) (.node 12451 .empty .empty)) (.node 14011 (.node 13309 .empty .empty) (.node 14851 .empty .empty))) (.node 18793 (.node 18013 (.node 17383 .empty .empty) (.node 18211 .empty .empty)) (.node 22129 (.node 22003 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3637_ordered : SortedNatTree.ordered row3637 = true := by decide
set_option maxRecDepth 4096 in
theorem row3637_values : SortedNatTree.values row3637 = Data.row3637 := rfl
theorem row3637_checked : SortedNatTree.contains row3637 = Data.row3637.contains := by
  rw [SortedNatTree.contains_eq_list row3637 row3637_ordered, row3637_values]
def row3643 : SortedNatTree.Tree := (.node 15361 (.node 8581 (.node 5869 (.node 4987 (.node 4561 (.node 4273 .empty .empty) .empty) (.node 5449 .empty .empty)) (.node 6199 (.node 6073 .empty .empty) (.node 6607 .empty .empty))) (.node 10831 (.node 8971 (.node 8929 .empty .empty) (.node 9817 .empty .empty)) (.node 12721 (.node 11587 .empty .empty) (.node 13687 .empty .empty)))) (.node 18553 (.node 16741 (.node 16231 (.node 15817 .empty .empty) (.node 16417 .empty .empty)) (.node 18127 (.node 17881 .empty .empty) (.node 18289 .empty .empty))) (.node 21031 (.node 19819 (.node 19687 .empty .empty) (.node 20107 .empty .empty)) (.node 21187 (.node 21157 .empty .empty) (.node 21313 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3643_ordered : SortedNatTree.ordered row3643 = true := by decide
set_option maxRecDepth 4096 in
theorem row3643_values : SortedNatTree.values row3643 = Data.row3643 := rfl
theorem row3643_checked : SortedNatTree.contains row3643 = Data.row3643.contains := by
  rw [SortedNatTree.contains_eq_list row3643 row3643_ordered, row3643_values]
def row3659 : SortedNatTree.Tree := (.node 11069 (.node 7703 (.node 5981 (.node 4157 (.node 4091 (.node 3803 .empty .empty) .empty) (.node 4643 (.node 4373 .empty .empty) .empty)) (.node 6449 (.node 6317 (.node 6047 .empty .empty) .empty) (.node 7229 .empty .empty))) (.node 9473 (.node 9221 (.node 8273 (.node 7829 .empty .empty) .empty) (.node 9341 .empty .empty)) (.node 9851 (.node 9767 (.node 9677 .empty .empty) .empty) (.node 10163 .empty .empty)))) (.node 14639 (.node 13697 (.node 12941 (.node 12479 (.node 12161 .empty .empty) .empty) (.node 13217 (.node 12983 .empty .empty) .empty)) (.node 14387 (.node 14057 (.node 13901 .empty .empty) .empty) (.node 14561 .empty .empty))) (.node 15443 (.node 14957 (.node 14897 (.node 14741 .empty .empty) .empty) (.node 15287 .empty .empty)) (.node 21407 (.node 20117 (.node 17729 .empty .empty) .empty) (.node 22079 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3659_ordered : SortedNatTree.ordered row3659 = true := by decide
set_option maxRecDepth 4096 in
theorem row3659_values : SortedNatTree.values row3659 = Data.row3659 := rfl
theorem row3659_checked : SortedNatTree.contains row3659 = Data.row3659.contains := by
  rw [SortedNatTree.contains_eq_list row3659 row3659_ordered, row3659_values]
def row3671 : SortedNatTree.Tree := (.node 9929 (.node 7523 (.node 6011 (.node 5381 (.node 5039 (.node 4091 .empty .empty) .empty) (.node 5849 .empty .empty)) (.node 7019 (.node 6653 (.node 6131 .empty .empty) .empty) (.node 7151 .empty .empty))) (.node 9323 (.node 8447 (.node 7853 (.node 7691 .empty .empty) .empty) (.node 8501 .empty .empty)) (.node 9623 (.node 9467 (.node 9461 .empty .empty) .empty) (.node 9923 .empty .empty)))) (.node 15461 (.node 12953 (.node 12413 (.node 10301 (.node 10181 .empty .empty) .empty) (.node 12437 .empty .empty)) (.node 14537 (.node 13883 (.node 13337 .empty .empty) .empty) (.node 15173 .empty .empty))) (.node 18287 (.node 17939 (.node 17483 (.node 17327 .empty .empty) .empty) (.node 18041 .empty .empty)) (.node 20123 (.node 19751 (.node 19073 .empty .empty) .empty) (.node 22307 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row3671_ordered : SortedNatTree.ordered row3671 = true := by decide
set_option maxRecDepth 4096 in
theorem row3671_values : SortedNatTree.values row3671 = Data.row3671 := rfl
theorem row3671_checked : SortedNatTree.contains row3671 = Data.row3671.contains := by
  rw [SortedNatTree.contains_eq_list row3671 row3671_ordered, row3671_values]
def row3673 : SortedNatTree.Tree := (.node 13183 (.node 7717 (.node 6529 (.node 4759 (.node 4129 .empty .empty) (.node 5581 .empty .empty)) (.node 7537 (.node 6691 .empty .empty) .empty)) (.node 9151 (.node 8629 (.node 8293 .empty .empty) .empty) (.node 11731 (.node 9421 .empty .empty) .empty))) (.node 18229 (.node 15307 (.node 14713 (.node 14407 .empty .empty) (.node 15259 .empty .empty)) (.node 17761 (.node 17053 .empty .empty) .empty)) (.node 20029 (.node 19501 (.node 18859 .empty .empty) .empty) (.node 21061 (.node 20389 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3673_ordered : SortedNatTree.ordered row3673 = true := by decide
set_option maxRecDepth 4096 in
theorem row3673_values : SortedNatTree.values row3673 = Data.row3673 := rfl
theorem row3673_checked : SortedNatTree.contains row3673 = Data.row3673.contains := by
  rw [SortedNatTree.contains_eq_list row3673 row3673_ordered, row3673_values]
def row3677 : SortedNatTree.Tree := (.node 12269 (.node 7583 (.node 5903 (.node 4733 (.node 4211 .empty .empty) (.node 5807 .empty .empty)) (.node 7229 (.node 6659 .empty .empty) .empty)) (.node 10151 (.node 9377 (.node 8069 .empty .empty) (.node 9629 .empty .empty)) (.node 11969 (.node 11321 .empty .empty) .empty))) (.node 17033 (.node 14411 (.node 13679 (.node 13613 .empty .empty) (.node 14327 .empty .empty)) (.node 16763 (.node 14813 .empty .empty) .empty)) (.node 19139 (.node 18401 (.node 17417 .empty .empty) (.node 18731 .empty .empty)) (.node 22259 (.node 21617 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row3677_ordered : SortedNatTree.ordered row3677 = true := by decide
set_option maxRecDepth 4096 in
theorem row3677_values : SortedNatTree.values row3677 = Data.row3677 := rfl
theorem row3677_checked : SortedNatTree.contains row3677 = Data.row3677.contains := by
  rw [SortedNatTree.contains_eq_list row3677 row3677_ordered, row3677_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
