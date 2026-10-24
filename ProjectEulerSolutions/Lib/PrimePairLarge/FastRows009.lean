import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row4229 : SortedNatTree.Tree := (.node 16073 (.node 8831 (.node 6689 (.node 5039 (.node 4391 .empty .empty) (.node 5399 .empty .empty)) (.node 7643 (.node 7589 .empty .empty) .empty)) (.node 11399 (.node 10457 (.node 9941 .empty .empty) (.node 11213 .empty .empty)) (.node 15683 (.node 11549 .empty .empty) .empty))) (.node 17669 (.node 16883 (.node 16607 (.node 16349 .empty .empty) (.node 16619 .empty .empty)) (.node 17597 (.node 17021 .empty .empty) .empty)) (.node 20219 (.node 18719 (.node 18371 .empty .empty) .empty) (.node 21221 (.node 20483 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4229_ordered : SortedNatTree.ordered row4229 = true := by decide
set_option maxRecDepth 4096 in
theorem row4229_values : SortedNatTree.values row4229 = Data.row4229 := rfl
theorem row4229_checked : SortedNatTree.contains row4229 = Data.row4229.contains := by
  rw [SortedNatTree.contains_eq_list row4229 row4229_ordered, row4229_values]
def row4231 : SortedNatTree.Tree := (.node 12253 (.node 7789 (.node 6151 (.node 5197 (.node 4933 (.node 4339 .empty .empty) .empty) (.node 6037 .empty .empty)) (.node 6577 (.node 6529 .empty .empty) (.node 7027 .empty .empty))) (.node 9769 (.node 9547 (.node 8707 (.node 8623 .empty .empty) .empty) (.node 9721 .empty .empty)) (.node 10789 (.node 10477 .empty .empty) (.node 11527 .empty .empty)))) (.node 14947 (.node 14011 (.node 13591 (.node 12967 (.node 12763 .empty .empty) .empty) (.node 13729 .empty .empty)) (.node 14551 (.node 14143 .empty .empty) (.node 14563 .empty .empty))) (.node 18379 (.node 17203 (.node 16921 (.node 15991 .empty .empty) .empty) (.node 18169 .empty .empty)) (.node 19687 (.node 18637 .empty .empty) (.node 21499 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4231_ordered : SortedNatTree.ordered row4231 = true := by decide
set_option maxRecDepth 4096 in
theorem row4231_values : SortedNatTree.values row4231 = Data.row4231 := rfl
theorem row4231_checked : SortedNatTree.contains row4231 = Data.row4231.contains := by
  rw [SortedNatTree.contains_eq_list row4231 row4231_ordered, row4231_values]
def row4241 : SortedNatTree.Tree := (.node 10289 (.node 7559 (.node 6053 (.node 4793 (.node 4751 .empty .empty) (.node 5981 .empty .empty)) (.node 6977 (.node 6389 .empty .empty) (.node 7103 .empty .empty))) (.node 8147 (.node 7757 (.node 7673 .empty .empty) (.node 8093 .empty .empty)) (.node 9413 (.node 9323 .empty .empty) (.node 10259 .empty .empty)))) (.node 16937 (.node 13709 (.node 11351 (.node 10949 .empty .empty) (.node 13457 .empty .empty)) (.node 15173 (.node 15161 .empty .empty) (.node 15233 .empty .empty))) (.node 20231 (.node 18743 (.node 18521 .empty .empty) (.node 19433 .empty .empty)) (.node 21011 (.node 20327 .empty .empty) (.node 21701 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4241_ordered : SortedNatTree.ordered row4241 = true := by decide
set_option maxRecDepth 4096 in
theorem row4241_values : SortedNatTree.values row4241 = Data.row4241 := rfl
theorem row4241_checked : SortedNatTree.contains row4241 = Data.row4241.contains := by
  rw [SortedNatTree.contains_eq_list row4241 row4241_ordered, row4241_values]
def row4243 : SortedNatTree.Tree := (.node 11197 (.node 6469 (.node 4759 (.node 4591 (.node 4423 (.node 4339 .empty .empty) .empty) (.node 4621 .empty .empty)) (.node 5527 (.node 5347 (.node 4831 .empty .empty) .empty) (.node 6451 .empty .empty))) (.node 7699 (.node 7207 (.node 6997 (.node 6553 .empty .empty) .empty) (.node 7669 .empty .empty)) (.node 10723 (.node 10657 (.node 8581 .empty .empty) .empty) (.node 11119 .empty .empty)))) (.node 15361 (.node 12613 (.node 12391 (.node 11491 (.node 11299 .empty .empty) .empty) (.node 12457 .empty .empty)) (.node 14461 (.node 13807 (.node 13063 .empty .empty) .empty) (.node 14797 .empty .empty))) (.node 19081 (.node 18913 (.node 17209 (.node 16267 .empty .empty) .empty) (.node 19009 .empty .empty)) (.node 20707 (.node 19867 (.node 19309 .empty .empty) .empty) (.node 20857 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4243_ordered : SortedNatTree.ordered row4243 = true := by decide
set_option maxRecDepth 4096 in
theorem row4243_values : SortedNatTree.values row4243 = Data.row4243 := rfl
theorem row4243_checked : SortedNatTree.contains row4243 = Data.row4243.contains := by
  rw [SortedNatTree.contains_eq_list row4243 row4243_ordered, row4243_values]
def row4253 : SortedNatTree.Tree := (.node 11489 (.node 7829 (.node 5657 (.node 4463 (.node 4397 .empty .empty) (.node 4937 .empty .empty)) (.node 6917 (.node 5939 .empty .empty) .empty)) (.node 9293 (.node 8849 (.node 8831 .empty .empty) .empty) (.node 9941 (.node 9521 .empty .empty) .empty))) (.node 15461 (.node 11867 (.node 11681 (.node 11519 .empty .empty) (.node 11813 .empty .empty)) (.node 13997 (.node 12101 .empty .empty) .empty)) (.node 19709 (.node 17099 (.node 16937 .empty .empty) .empty) (.node 21599 (.node 20543 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4253_ordered : SortedNatTree.ordered row4253 = true := by decide
set_option maxRecDepth 4096 in
theorem row4253_values : SortedNatTree.values row4253 = Data.row4253 := rfl
theorem row4253_checked : SortedNatTree.contains row4253 = Data.row4253.contains := by
  rw [SortedNatTree.contains_eq_list row4253 row4253_ordered, row4253_values]
def row4259 : SortedNatTree.Tree := (.node 11909 (.node 9677 (.node 5807 (.node 5087 (.node 4787 .empty .empty) (.node 5417 .empty .empty)) (.node 8609 (.node 6221 .empty .empty) (.node 9551 .empty .empty))) (.node 10847 (.node 10181 (.node 10139 .empty .empty) (.node 10427 .empty .empty)) (.node 11597 (.node 11399 .empty .empty) .empty))) (.node 17351 (.node 15797 (.node 12953 (.node 12743 .empty .empty) (.node 15383 .empty .empty)) (.node 15887 (.node 15881 .empty .empty) .empty)) (.node 19001 (.node 18329 (.node 17729 .empty .empty) (.node 18443 .empty .empty)) (.node 21227 (.node 19709 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4259_ordered : SortedNatTree.ordered row4259 = true := by decide
set_option maxRecDepth 4096 in
theorem row4259_values : SortedNatTree.values row4259 = Data.row4259 := rfl
theorem row4259_checked : SortedNatTree.contains row4259 = Data.row4259.contains := by
  rw [SortedNatTree.contains_eq_list row4259 row4259_ordered, row4259_values]
def row4261 : SortedNatTree.Tree := (.node 12433 (.node 9319 (.node 7621 (.node 5281 (.node 5011 .empty .empty) (.node 5827 .empty .empty)) (.node 8179 (.node 7951 .empty .empty) .empty)) (.node 11437 (.node 9817 (.node 9397 .empty .empty) .empty) (.node 12073 (.node 11971 .empty .empty) .empty))) (.node 17737 (.node 15373 (.node 13267 (.node 12451 .empty .empty) (.node 14869 .empty .empty)) (.node 16843 (.node 15787 .empty .empty) .empty)) (.node 20719 (.node 19867 (.node 18541 .empty .empty) .empty) (.node 21019 (.node 20743 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4261_ordered : SortedNatTree.ordered row4261 = true := by decide
set_option maxRecDepth 4096 in
theorem row4261_values : SortedNatTree.values row4261 = Data.row4261 := rfl
theorem row4261_checked : SortedNatTree.contains row4261 = Data.row4261.contains := by
  rw [SortedNatTree.contains_eq_list row4261 row4261_ordered, row4261_values]
def row4271 : SortedNatTree.Tree := (.node 12659 (.node 9029 (.node 5693 (.node 5009 (.node 4673 (.node 4517 .empty .empty) .empty) (.node 5573 .empty .empty)) (.node 7823 (.node 6857 (.node 6131 .empty .empty) .empty) (.node 8693 .empty .empty))) (.node 10427 (.node 10037 (.node 9491 (.node 9137 .empty .empty) .empty) (.node 10247 .empty .empty)) (.node 11777 (.node 11549 .empty .empty) (.node 12113 .empty .empty)))) (.node 17021 (.node 15443 (.node 14591 (.node 14489 (.node 12809 .empty .empty) .empty) (.node 15077 .empty .empty)) (.node 16691 (.node 16451 (.node 16127 .empty .empty) .empty) (.node 16883 .empty .empty))) (.node 20063 (.node 18701 (.node 18617 (.node 17189 .empty .empty) .empty) (.node 19211 .empty .empty)) (.node 20663 (.node 20549 .empty .empty) (.node 21347 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4271_ordered : SortedNatTree.ordered row4271 = true := by decide
set_option maxRecDepth 4096 in
theorem row4271_values : SortedNatTree.values row4271 = Data.row4271 := rfl
theorem row4271_checked : SortedNatTree.contains row4271 = Data.row4271.contains := by
  rw [SortedNatTree.contains_eq_list row4271 row4271_ordered, row4271_values]
def row4273 : SortedNatTree.Tree := (.node 9883 (.node 7489 (.node 6247 (.node 4969 (.node 4723 (.node 4567 .empty .empty) .empty) (.node 5023 .empty .empty)) (.node 7207 (.node 6703 (.node 6373 .empty .empty) .empty) (.node 7297 .empty .empty))) (.node 8461 (.node 7951 (.node 7573 (.node 7549 .empty .empty) .empty) (.node 8287 .empty .empty)) (.node 9007 (.node 8803 .empty .empty) (.node 9397 .empty .empty)))) (.node 15991 (.node 12703 (.node 10867 (.node 10567 (.node 10243 .empty .empty) .empty) (.node 12421 .empty .empty)) (.node 14683 (.node 13171 .empty .empty) (.node 15277 .empty .empty))) (.node 20743 (.node 19609 (.node 17497 (.node 16921 .empty .empty) .empty) (.node 20731 .empty .empty)) (.node 21157 (.node 20809 .empty .empty) (.node 21277 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4273_ordered : SortedNatTree.ordered row4273 = true := by decide
set_option maxRecDepth 4096 in
theorem row4273_values : SortedNatTree.values row4273 = Data.row4273 := rfl
theorem row4273_checked : SortedNatTree.contains row4273 = Data.row4273.contains := by
  rw [SortedNatTree.contains_eq_list row4273 row4273_ordered, row4273_values]
def row4283 : SortedNatTree.Tree := (.node 14423 (.node 8273 (.node 7103 (.node 4523 (.node 4289 .empty .empty) (.node 5669 .empty .empty)) (.node 8093 (.node 8069 .empty .empty) .empty)) (.node 10343 (.node 9833 (.node 8447 .empty .empty) .empty) (.node 13241 (.node 10613 .empty .empty) .empty))) (.node 18287 (.node 16901 (.node 15359 (.node 14657 .empty .empty) .empty) (.node 18089 (.node 17471 .empty .empty) .empty)) (.node 19421 (.node 19391 (.node 19379 .empty .empty) .empty) (.node 21701 (.node 20483 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4283_ordered : SortedNatTree.ordered row4283 = true := by decide
set_option maxRecDepth 4096 in
theorem row4283_values : SortedNatTree.values row4283 = Data.row4283 := rfl
theorem row4283_checked : SortedNatTree.contains row4283 = Data.row4283.contains := by
  rw [SortedNatTree.contains_eq_list row4283 row4283_ordered, row4283_values]
def row4289 : SortedNatTree.Tree := (.node 11471 (.node 7937 (.node 5519 (.node 4889 (.node 4679 (.node 4391 .empty .empty) .empty) (.node 5507 .empty .empty)) (.node 6869 (.node 6473 (.node 6317 .empty .empty) .empty) (.node 7331 .empty .empty))) (.node 9803 (.node 8819 (.node 8171 (.node 8087 .empty .empty) .empty) (.node 8861 .empty .empty)) (.node 10709 (.node 10211 .empty .empty) (.node 11159 .empty .empty)))) (.node 16091 (.node 13151 (.node 11867 (.node 11831 (.node 11489 .empty .empty) .empty) (.node 11933 .empty .empty)) (.node 15737 (.node 14033 (.node 13721 .empty .empty) .empty) (.node 16073 .empty .empty))) (.node 20477 (.node 18233 (.node 17387 (.node 16427 .empty .empty) .empty) (.node 19949 .empty .empty)) (.node 20903 (.node 20753 .empty .empty) (.node 21401 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4289_ordered : SortedNatTree.ordered row4289 = true := by decide
set_option maxRecDepth 4096 in
theorem row4289_values : SortedNatTree.values row4289 = Data.row4289 := rfl
theorem row4289_checked : SortedNatTree.contains row4289 = Data.row4289.contains := by
  rw [SortedNatTree.contains_eq_list row4289 row4289_ordered, row4289_values]
def row4297 : SortedNatTree.Tree := (.node 9619 (.node 6781 (.node 5521 (.node 4483 (.node 4441 .empty .empty) (.node 4507 .empty .empty)) (.node 6343 (.node 5689 .empty .empty) (.node 6607 .empty .empty))) (.node 8581 (.node 8233 (.node 7213 .empty .empty) (.node 8317 .empty .empty)) (.node 9241 (.node 9133 .empty .empty) .empty))) (.node 13693 (.node 11593 (.node 10111 (.node 9967 .empty .empty) (.node 11497 .empty .empty)) (.node 12697 (.node 12541 .empty .empty) (.node 13591 .empty .empty))) (.node 16747 (.node 14869 (.node 14011 .empty .empty) (.node 15259 .empty .empty)) (.node 19681 (.node 18379 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4297_ordered : SortedNatTree.ordered row4297 = true := by decide
set_option maxRecDepth 4096 in
theorem row4297_values : SortedNatTree.values row4297 = Data.row4297 := rfl
theorem row4297_checked : SortedNatTree.contains row4297 = Data.row4297.contains := by
  rw [SortedNatTree.contains_eq_list row4297 row4297_ordered, row4297_values]
def row4327 : SortedNatTree.Tree := (.node 11551 (.node 7639 (.node 5503 (.node 4801 (.node 4591 (.node 4519 .empty .empty) .empty) (.node 5107 .empty .empty)) (.node 6553 (.node 6481 (.node 5737 .empty .empty) .empty) (.node 6703 .empty .empty))) (.node 9619 (.node 9241 (.node 8233 (.node 7963 .empty .empty) .empty) (.node 9337 .empty .empty)) (.node 9883 (.node 9871 (.node 9697 .empty .empty) .empty) (.node 10357 .empty .empty)))) (.node 17011 (.node 14251 (.node 12967 (.node 12109 (.node 12097 .empty .empty) .empty) (.node 13063 .empty .empty)) (.node 15919 (.node 15193 (.node 14389 .empty .empty) .empty) (.node 16069 .empty .empty))) (.node 19051 (.node 17539 (.node 17137 (.node 17107 .empty .empty) .empty) (.node 17749 .empty .empty)) (.node 20593 (.node 19387 .empty .empty) (.node 21277 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4327_ordered : SortedNatTree.ordered row4327 = true := by decide
set_option maxRecDepth 4096 in
theorem row4327_values : SortedNatTree.values row4327 = Data.row4327 := rfl
theorem row4327_checked : SortedNatTree.contains row4327 = Data.row4327.contains := by
  rw [SortedNatTree.contains_eq_list row4327 row4327_ordered, row4327_values]
def row4337 : SortedNatTree.Tree := (.node 12269 (.node 7043 (.node 5099 (.node 4523 (.node 4421 .empty .empty) (.node 4889 .empty .empty)) (.node 5471 (.node 5237 .empty .empty) .empty)) (.node 9551 (.node 9293 (.node 7547 .empty .empty) .empty) (.node 11903 (.node 11321 .empty .empty) .empty))) (.node 18251 (.node 17183 (.node 15473 (.node 13613 .empty .empty) .empty) (.node 18119 (.node 17981 .empty .empty) .empty)) (.node 20921 (.node 19037 (.node 18371 .empty .empty) .empty) (.node 21401 (.node 21107 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4337_ordered : SortedNatTree.ordered row4337 = true := by decide
set_option maxRecDepth 4096 in
theorem row4337_values : SortedNatTree.values row4337 = Data.row4337 := rfl
theorem row4337_checked : SortedNatTree.contains row4337 = Data.row4337.contains := by
  rw [SortedNatTree.contains_eq_list row4337 row4337_ordered, row4337_values]
def row4339 : SortedNatTree.Tree := (.node 11329 (.node 8461 (.node 6961 (.node 4831 (.node 4639 (.node 4567 .empty .empty) .empty) (.node 5701 (.node 4903 .empty .empty) .empty)) (.node 8167 (.node 7927 (.node 7603 .empty .empty) .empty) (.node 8317 .empty .empty))) (.node 10303 (.node 9679 (.node 9661 (.node 9241 .empty .empty) .empty) (.node 9931 .empty .empty)) (.node 11047 (.node 10903 (.node 10651 .empty .empty) .empty) (.node 11131 .empty .empty)))) (.node 16657 (.node 13681 (.node 12907 (.node 12421 (.node 12379 .empty .empty) .empty) (.node 13567 .empty .empty)) (.node 14563 (.node 14479 (.node 14083 .empty .empty) .empty) (.node 16369 .empty .empty))) (.node 20101 (.node 19309 (.node 18787 (.node 17431 .empty .empty) .empty) (.node 19417 .empty .empty)) (.node 21559 (.node 21193 (.node 20551 .empty .empty) .empty) (.node 21613 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4339_ordered : SortedNatTree.ordered row4339 = true := by decide
set_option maxRecDepth 4096 in
theorem row4339_values : SortedNatTree.values row4339 = Data.row4339 := rfl
theorem row4339_checked : SortedNatTree.contains row4339 = Data.row4339.contains := by
  rw [SortedNatTree.contains_eq_list row4339 row4339_ordered, row4339_values]
def row4349 : SortedNatTree.Tree := (.node 12227 (.node 7793 (.node 5531 (.node 5297 (.node 5051 (.node 4967 .empty .empty) .empty) (.node 5483 (.node 5303 .empty .empty) .empty)) (.node 7019 (.node 6581 (.node 5861 .empty .empty) .empty) (.node 7523 .empty .empty))) (.node 10607 (.node 9629 (.node 9041 (.node 8093 .empty .empty) .empty) (.node 10313 .empty .empty)) (.node 11177 (.node 11171 (.node 11027 .empty .empty) .empty) (.node 12119 .empty .empty)))) (.node 16301 (.node 13697 (.node 13001 (.node 12893 (.node 12713 .empty .empty) .empty) (.node 13217 .empty .empty)) (.node 14411 (.node 14321 (.node 13967 .empty .empty) .empty) (.node 15629 .empty .empty))) (.node 19889 (.node 18233 (.node 17471 (.node 17207 .empty .empty) .empty) (.node 19427 .empty .empty)) (.node 21089 (.node 20873 (.node 20663 .empty .empty) .empty) (.node 21107 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4349_ordered : SortedNatTree.ordered row4349 = true := by decide
set_option maxRecDepth 4096 in
theorem row4349_values : SortedNatTree.values row4349 = Data.row4349 := rfl
theorem row4349_checked : SortedNatTree.contains row4349 = Data.row4349.contains := by
  rw [SortedNatTree.contains_eq_list row4349 row4349_ordered, row4349_values]
def row4357 : SortedNatTree.Tree := (.node 14341 (.node 7297 (.node 5479 (.node 5107 (.node 5101 (.node 4813 .empty .empty) .empty) (.node 5449 (.node 5431 .empty .empty) .empty)) (.node 6691 (.node 6619 (.node 5563 .empty .empty) .empty) (.node 6829 .empty .empty))) (.node 9859 (.node 8209 (.node 8053 (.node 7411 .empty .empty) .empty) (.node 9001 (.node 8893 .empty .empty) .empty)) (.node 12973 (.node 12799 (.node 11491 .empty .empty) .empty) (.node 13183 .empty .empty)))) (.node 18043 (.node 16693 (.node 15739 (.node 15451 (.node 14551 .empty .empty) .empty) (.node 16573 (.node 15901 .empty .empty) .empty)) (.node 17749 (.node 17383 (.node 17167 .empty .empty) .empty) (.node 17959 .empty .empty))) (.node 19813 (.node 18289 (.node 18133 (.node 18121 .empty .empty) .empty) (.node 18553 (.node 18307 .empty .empty) .empty)) (.node 21487 (.node 21397 (.node 19861 .empty .empty) .empty) (.node 21613 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4357_ordered : SortedNatTree.ordered row4357 = true := by decide
set_option maxRecDepth 4096 in
theorem row4357_values : SortedNatTree.values row4357 = Data.row4357 := rfl
theorem row4357_checked : SortedNatTree.contains row4357 = Data.row4357.contains := by
  rw [SortedNatTree.contains_eq_list row4357 row4357_ordered, row4357_values]
def row4363 : SortedNatTree.Tree := (.node 13147 (.node 8761 (.node 7621 (.node 6133 (.node 5653 (.node 4813 .empty .empty) .empty) (.node 7459 .empty .empty)) (.node 8461 (.node 8311 (.node 7687 .empty .empty) .empty) (.node 8713 .empty .empty))) (.node 10453 (.node 9319 (.node 9199 (.node 9091 .empty .empty) .empty) (.node 10009 .empty .empty)) (.node 11779 (.node 11527 .empty .empty) (.node 13063 .empty .empty)))) (.node 16729 (.node 14869 (.node 13873 (.node 13417 (.node 13291 .empty .empty) .empty) (.node 14779 .empty .empty)) (.node 16063 (.node 15889 (.node 14929 .empty .empty) .empty) (.node 16267 .empty .empty))) (.node 19423 (.node 17137 (.node 17029 (.node 16963 .empty .empty) .empty) (.node 18367 .empty .empty)) (.node 20719 (.node 19801 .empty .empty) (.node 20743 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4363_ordered : SortedNatTree.ordered row4363 = true := by decide
set_option maxRecDepth 4096 in
theorem row4363_values : SortedNatTree.values row4363 = Data.row4363 := rfl
theorem row4363_checked : SortedNatTree.contains row4363 = Data.row4363.contains := by
  rw [SortedNatTree.contains_eq_list row4363 row4363_ordered, row4363_values]
def row4373 : SortedNatTree.Tree := (.node 10487 (.node 8039 (.node 5927 (.node 5693 (.node 5297 (.node 5189 .empty .empty) .empty) (.node 5801 .empty .empty)) (.node 7541 (.node 7121 (.node 7019 .empty .empty) .empty) (.node 7919 .empty .empty))) (.node 9431 (.node 8423 (.node 8237 (.node 8147 .empty .empty) .empty) (.node 8609 .empty .empty)) (.node 10163 (.node 9551 .empty .empty) (.node 10463 .empty .empty)))) (.node 14747 (.node 12377 (.node 11783 (.node 11471 (.node 10739 .empty .empty) .empty) (.node 11807 .empty .empty)) (.node 12983 (.node 12791 (.node 12497 .empty .empty) .empty) (.node 14321 .empty .empty))) (.node 18899 (.node 16883 (.node 16631 (.node 15287 .empty .empty) .empty) (.node 17159 .empty .empty)) (.node 20249 (.node 19853 .empty .empty) (.node 20981 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4373_ordered : SortedNatTree.ordered row4373 = true := by decide
set_option maxRecDepth 4096 in
theorem row4373_values : SortedNatTree.values row4373 = Data.row4373 := rfl
theorem row4373_checked : SortedNatTree.contains row4373 = Data.row4373.contains := by
  rw [SortedNatTree.contains_eq_list row4373 row4373_ordered, row4373_values]
def row4391 : SortedNatTree.Tree := (.node 12497 (.node 8783 (.node 7901 (.node 6263 (.node 5639 .empty .empty) (.node 6869 .empty .empty)) (.node 8219 (.node 8111 .empty .empty) .empty)) (.node 10499 (.node 9539 (.node 8999 .empty .empty) (.node 10211 .empty .empty)) (.node 12203 (.node 11579 .empty .empty) .empty))) (.node 16361 (.node 14831 (.node 13397 (.node 12647 .empty .empty) (.node 14813 .empty .empty)) (.node 16193 (.node 14867 .empty .empty) .empty)) (.node 18587 (.node 18521 (.node 17519 .empty .empty) .empty) (.node 21383 (.node 19997 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4391_ordered : SortedNatTree.ordered row4391 = true := by decide
set_option maxRecDepth 4096 in
theorem row4391_values : SortedNatTree.values row4391 = Data.row4391 := rfl
theorem row4391_checked : SortedNatTree.contains row4391 = Data.row4391.contains := by
  rw [SortedNatTree.contains_eq_list row4391 row4391_ordered, row4391_values]
def row4397 : SortedNatTree.Tree := (.node 11549 (.node 7823 (.node 6311 (.node 4703 (.node 4691 (.node 4421 .empty .empty) .empty) (.node 4871 .empty .empty)) (.node 6833 (.node 6701 (.node 6563 .empty .empty) .empty) (.node 7013 .empty .empty))) (.node 9227 (.node 8513 (.node 8087 (.node 7841 .empty .empty) .empty) (.node 8861 .empty .empty)) (.node 11057 (.node 10487 (.node 9431 .empty .empty) .empty) (.node 11411 .empty .empty)))) (.node 16253 (.node 14813 (.node 12809 (.node 12473 (.node 12149 .empty .empty) .empty) (.node 13121 .empty .empty)) (.node 16097 (.node 15791 (.node 15641 .empty .empty) .empty) (.node 16127 .empty .empty))) (.node 19571 (.node 18959 (.node 17123 (.node 16451 .empty .empty) .empty) (.node 19403 .empty .empty)) (.node 20219 (.node 19919 .empty .empty) (.node 21221 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4397_ordered : SortedNatTree.ordered row4397 = true := by decide
set_option maxRecDepth 4096 in
theorem row4397_values : SortedNatTree.values row4397 = Data.row4397 := rfl
theorem row4397_checked : SortedNatTree.contains row4397 = Data.row4397.contains := by
  rw [SortedNatTree.contains_eq_list row4397 row4397_ordered, row4397_values]
def row4409 : SortedNatTree.Tree := (.node 12611 (.node 9059 (.node 6719 (.node 5477 (.node 5279 (.node 5087 .empty .empty) .empty) (.node 5867 .empty .empty)) (.node 7253 (.node 6863 (.node 6737 .empty .empty) .empty) (.node 8447 .empty .empty))) (.node 9791 (.node 9413 (.node 9371 (.node 9239 .empty .empty) .empty) (.node 9689 .empty .empty)) (.node 11483 (.node 10937 (.node 10259 .empty .empty) .empty) (.node 11987 .empty .empty)))) (.node 16253 (.node 14153 (.node 13523 (.node 13469 (.node 13331 .empty .empty) .empty) (.node 13997 .empty .empty)) (.node 16061 (.node 15161 (.node 14657 .empty .empty) .empty) (.node 16103 .empty .empty))) (.node 18743 (.node 18521 (.node 18311 (.node 18287 .empty .empty) .empty) (.node 18719 .empty .empty)) (.node 20021 (.node 19793 (.node 19709 .empty .empty) .empty) (.node 21347 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4409_ordered : SortedNatTree.ordered row4409 = true := by decide
set_option maxRecDepth 4096 in
theorem row4409_values : SortedNatTree.values row4409 = Data.row4409 := rfl
theorem row4409_checked : SortedNatTree.contains row4409 = Data.row4409.contains := by
  rw [SortedNatTree.contains_eq_list row4409 row4409_ordered, row4409_values]
def row4421 : SortedNatTree.Tree := (.node 11471 (.node 9587 (.node 6869 (.node 6563 (.node 6113 (.node 5783 .empty .empty) .empty) (.node 6791 .empty .empty)) (.node 7673 (.node 7577 (.node 6899 .empty .empty) .empty) (.node 8039 .empty .empty))) (.node 10631 (.node 10139 (.node 9887 (.node 9803 .empty .empty) .empty) (.node 10463 .empty .empty)) (.node 11279 (.node 10883 .empty .empty) (.node 11393 .empty .empty)))) (.node 16931 (.node 14243 (.node 13163 (.node 12401 (.node 11597 .empty .empty) .empty) (.node 13241 .empty .empty)) (.node 15803 (.node 15629 (.node 15101 .empty .empty) .empty) (.node 16427 .empty .empty))) (.node 19661 (.node 18671 (.node 18617 (.node 18077 .empty .empty) .empty) (.node 18947 .empty .empty)) (.node 20759 (.node 19763 .empty .empty) (.node 20771 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4421_ordered : SortedNatTree.ordered row4421 = true := by decide
set_option maxRecDepth 4096 in
theorem row4421_values : SortedNatTree.values row4421 = Data.row4421 := rfl
theorem row4421_checked : SortedNatTree.contains row4421 = Data.row4421.contains := by
  rw [SortedNatTree.contains_eq_list row4421 row4421_ordered, row4421_values]
def row4423 : SortedNatTree.Tree := (.node 12757 (.node 9343 (.node 6133 (.node 5689 (.node 5449 .empty .empty) (.node 5869 .empty .empty)) (.node 8863 (.node 8317 .empty .empty) (.node 9337 .empty .empty))) (.node 10357 (.node 9817 (.node 9421 .empty .empty) (.node 10303 .empty .empty)) (.node 12109 (.node 10567 .empty .empty) .empty))) (.node 16141 (.node 15187 (.node 13537 (.node 13411 .empty .empty) (.node 13693 .empty .empty)) (.node 15901 (.node 15823 .empty .empty) (.node 16087 .empty .empty))) (.node 20719 (.node 17431 (.node 17011 .empty .empty) (.node 19333 .empty .empty)) (.node 21013 (.node 20887 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4423_ordered : SortedNatTree.ordered row4423 = true := by decide
set_option maxRecDepth 4096 in
theorem row4423_values : SortedNatTree.values row4423 = Data.row4423 := rfl
theorem row4423_checked : SortedNatTree.contains row4423 = Data.row4423.contains := by
  rw [SortedNatTree.contains_eq_list row4423 row4423_ordered, row4423_values]
def row4441 : SortedNatTree.Tree := (.node 7759 (.node 6271 (.node 5743 (.node 5347 (.node 5227 .empty .empty) .empty) (.node 6247 (.node 5953 .empty .empty) .empty)) (.node 6481 (.node 6373 (.node 6367 .empty .empty) .empty) (.node 6949 .empty .empty))) (.node 12007 (.node 9127 (.node 8761 (.node 7873 .empty .empty) .empty) (.node 10477 (.node 9547 .empty .empty) .empty)) (.node 21319 (.node 21247 (.node 17497 .empty .empty) .empty) (.node 21577 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row4441_ordered : SortedNatTree.ordered row4441 = true := by decide
set_option maxRecDepth 4096 in
theorem row4441_values : SortedNatTree.values row4441 = Data.row4441 := rfl
theorem row4441_checked : SortedNatTree.contains row4441 = Data.row4441.contains := by
  rw [SortedNatTree.contains_eq_list row4441 row4441_ordered, row4441_values]
def row4447 : SortedNatTree.Tree := (.node 12553 (.node 8101 (.node 6067 (.node 5023 (.node 4789 .empty .empty) (.node 5419 .empty .empty)) (.node 7057 (.node 6427 .empty .empty) (.node 7207 .empty .empty))) (.node 9811 (.node 8839 (.node 8293 .empty .empty) (.node 8923 .empty .empty)) (.node 11677 (.node 10831 .empty .empty) (.node 11941 .empty .empty)))) (.node 18289 (.node 15259 (.node 14419 (.node 12703 .empty .empty) (.node 15031 .empty .empty)) (.node 17761 (.node 17239 .empty .empty) (.node 17989 .empty .empty))) (.node 19441 (.node 18787 (.node 18517 .empty .empty) (.node 19069 .empty .empty)) (.node 20407 (.node 20047 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4447_ordered : SortedNatTree.ordered row4447 = true := by decide
set_option maxRecDepth 4096 in
theorem row4447_values : SortedNatTree.values row4447 = Data.row4447 := rfl
theorem row4447_checked : SortedNatTree.contains row4447 = Data.row4447.contains := by
  rw [SortedNatTree.contains_eq_list row4447 row4447_ordered, row4447_values]
def row4451 : SortedNatTree.Tree := (.node 11789 (.node 8681 (.node 6803 (.node 6257 (.node 6089 (.node 5279 .empty .empty) .empty) (.node 6599 .empty .empty)) (.node 7307 (.node 6833 .empty .empty) (.node 7853 .empty .empty))) (.node 9521 (.node 8861 (.node 8783 .empty .empty) (.node 8963 .empty .empty)) (.node 9791 (.node 9533 .empty .empty) (.node 11483 .empty .empty)))) (.node 16943 (.node 14753 (.node 14057 (.node 13103 .empty .empty) (.node 14669 .empty .empty)) (.node 15791 (.node 15053 .empty .empty) (.node 16103 .empty .empty))) (.node 19937 (.node 17909 (.node 17471 .empty .empty) (.node 18911 .empty .empty)) (.node 20873 (.node 20369 .empty .empty) (.node 21503 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4451_ordered : SortedNatTree.ordered row4451 = true := by decide
set_option maxRecDepth 4096 in
theorem row4451_values : SortedNatTree.values row4451 = Data.row4451 := rfl
theorem row4451_checked : SortedNatTree.contains row4451 = Data.row4451.contains := by
  rw [SortedNatTree.contains_eq_list row4451 row4451_ordered, row4451_values]
def row4457 : SortedNatTree.Tree := (.node 9791 (.node 7331 (.node 5591 (.node 5393 (.node 4793 (.node 4679 .empty .empty) .empty) (.node 5507 (.node 5483 .empty .empty) .empty)) (.node 6491 (.node 6473 (.node 5981 .empty .empty) .empty) (.node 7019 .empty .empty))) (.node 7919 (.node 7607 (.node 7499 (.node 7433 .empty .empty) .empty) (.node 7853 .empty .empty)) (.node 9239 (.node 9203 (.node 8543 .empty .empty) .empty) (.node 9281 .empty .empty)))) (.node 15791 (.node 12809 (.node 10313 (.node 10163 (.node 10061 .empty .empty) .empty) (.node 12413 (.node 11579 .empty .empty) .empty)) (.node 14561 (.node 13487 (.node 13007 .empty .empty) .empty) (.node 15773 .empty .empty))) (.node 18089 (.node 17669 (.node 17579 (.node 16547 .empty .empty) .empty) (.node 17729 .empty .empty)) (.node 20771 (.node 19997 (.node 18149 .empty .empty) .empty) (.node 21407 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4457_ordered : SortedNatTree.ordered row4457 = true := by decide
set_option maxRecDepth 4096 in
theorem row4457_values : SortedNatTree.values row4457 = Data.row4457 := rfl
theorem row4457_checked : SortedNatTree.contains row4457 = Data.row4457.contains := by
  rw [SortedNatTree.contains_eq_list row4457 row4457_ordered, row4457_values]
def row4463 : SortedNatTree.Tree := (.node 12107 (.node 7877 (.node 5861 (.node 5441 (.node 5399 (.node 5273 .empty .empty) .empty) (.node 5813 .empty .empty)) (.node 6791 (.node 6521 .empty .empty) (.node 7079 .empty .empty))) (.node 9851 (.node 8963 (.node 8387 (.node 8369 .empty .empty) .empty) (.node 9587 .empty .empty)) (.node 10859 (.node 10301 .empty .empty) (.node 11933 .empty .empty)))) (.node 17483 (.node 13613 (.node 12491 (.node 12281 (.node 12143 .empty .empty) .empty) (.node 13421 .empty .empty)) (.node 15383 (.node 14411 .empty .empty) (.node 16319 .empty .empty))) (.node 18839 (.node 18077 (.node 17957 .empty .empty) (.node 18701 .empty .empty)) (.node 19157 (.node 18947 .empty .empty) (.node 20477 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4463_ordered : SortedNatTree.ordered row4463 = true := by decide
set_option maxRecDepth 4096 in
theorem row4463_values : SortedNatTree.values row4463 = Data.row4463 := rfl
theorem row4463_checked : SortedNatTree.contains row4463 = Data.row4463.contains := by
  rw [SortedNatTree.contains_eq_list row4463 row4463_ordered, row4463_values]
def row4481 : SortedNatTree.Tree := (.node 12641 (.node 7907 (.node 6791 (.node 6203 (.node 5189 .empty .empty) (.node 6491 .empty .empty)) (.node 7529 (.node 7193 .empty .empty) (.node 7583 .empty .empty))) (.node 10139 (.node 8831 (.node 8117 .empty .empty) (.node 9743 .empty .empty)) (.node 11003 (.node 10847 .empty .empty) (.node 11423 .empty .empty)))) (.node 17231 (.node 15401 (.node 14657 (.node 13397 .empty .empty) (.node 15329 .empty .empty)) (.node 15809 (.node 15749 .empty .empty) (.node 16979 .empty .empty))) (.node 20297 (.node 19391 (.node 18329 .empty .empty) (.node 19583 .empty .empty)) (.node 21269 (.node 21227 .empty .empty) (.node 21491 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4481_ordered : SortedNatTree.ordered row4481 = true := by decide
set_option maxRecDepth 4096 in
theorem row4481_values : SortedNatTree.values row4481 = Data.row4481 := rfl
theorem row4481_checked : SortedNatTree.contains row4481 = Data.row4481.contains := by
  rw [SortedNatTree.contains_eq_list row4481 row4481_ordered, row4481_values]
def row4483 : SortedNatTree.Tree := (.node 9049 (.node 6793 (.node 5737 (.node 5347 (.node 5113 .empty .empty) .empty) (.node 6607 (.node 6577 .empty .empty) .empty)) (.node 7717 (.node 7537 (.node 7489 .empty .empty) .empty) (.node 7759 .empty .empty))) (.node 17959 (.node 12289 (.node 11923 (.node 9463 .empty .empty) .empty) (.node 14221 (.node 13963 .empty .empty) .empty)) (.node 21319 (.node 20407 (.node 20011 .empty .empty) .empty) (.node 21493 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row4483_ordered : SortedNatTree.ordered row4483 = true := by decide
set_option maxRecDepth 4096 in
theorem row4483_values : SortedNatTree.values row4483 = Data.row4483 := rfl
theorem row4483_checked : SortedNatTree.contains row4483 = Data.row4483.contains := by
  rw [SortedNatTree.contains_eq_list row4483 row4483_ordered, row4483_values]
def row4493 : SortedNatTree.Tree := (.node 11777 (.node 7187 (.node 5519 (.node 5381 (.node 4793 .empty .empty) (.node 5507 .empty .empty)) (.node 6053 (.node 5717 .empty .empty) (.node 6173 .empty .empty))) (.node 10151 (.node 8693 (.node 7349 .empty .empty) (.node 10091 .empty .empty)) (.node 10889 (.node 10529 .empty .empty) (.node 11243 .empty .empty)))) (.node 17669 (.node 15497 (.node 14741 (.node 13499 .empty .empty) (.node 14843 .empty .empty)) (.node 17027 (.node 15641 .empty .empty) (.node 17597 .empty .empty))) (.node 20903 (.node 19991 (.node 18773 .empty .empty) (.node 20297 .empty .empty)) (.node 21221 (.node 21143 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4493_ordered : SortedNatTree.ordered row4493 = true := by decide
set_option maxRecDepth 4096 in
theorem row4493_values : SortedNatTree.values row4493 = Data.row4493 := rfl
theorem row4493_checked : SortedNatTree.contains row4493 = Data.row4493.contains := by
  rw [SortedNatTree.contains_eq_list row4493 row4493_ordered, row4493_values]
def row4507 : SortedNatTree.Tree := (.node 12421 (.node 7759 (.node 6091 (.node 4999 (.node 4801 (.node 4789 .empty .empty) .empty) (.node 5881 (.node 5419 .empty .empty) .empty)) (.node 7129 (.node 6709 (.node 6217 .empty .empty) .empty) (.node 7603 .empty .empty))) (.node 9901 (.node 8893 (.node 8377 (.node 7993 .empty .empty) .empty) (.node 9187 .empty .empty)) (.node 12253 (.node 11257 (.node 9967 .empty .empty) .empty) (.node 12277 .empty .empty)))) (.node 16267 (.node 14533 (.node 13327 (.node 12907 (.node 12739 .empty .empty) .empty) (.node 13807 (.node 13633 .empty .empty) .empty)) (.node 15439 (.node 15277 (.node 15217 .empty .empty) .empty) (.node 15919 .empty .empty))) (.node 18451 (.node 18061 (.node 17011 (.node 16759 .empty .empty) .empty) (.node 18211 .empty .empty)) (.node 21319 (.node 21283 (.node 19309 .empty .empty) .empty) (.node 21433 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4507_ordered : SortedNatTree.ordered row4507 = true := by decide
set_option maxRecDepth 4096 in
theorem row4507_values : SortedNatTree.values row4507 = Data.row4507 := rfl
theorem row4507_checked : SortedNatTree.contains row4507 = Data.row4507.contains := by
  rw [SortedNatTree.contains_eq_list row4507 row4507_ordered, row4507_values]
def row4513 : SortedNatTree.Tree := (.node 10597 (.node 8707 (.node 6277 (.node 5581 (.node 5437 (.node 4591 .empty .empty) .empty) (.node 6163 .empty .empty)) (.node 7723 (.node 6841 (.node 6397 .empty .empty) .empty) (.node 8287 .empty .empty))) (.node 9697 (.node 9187 (.node 9151 (.node 9049 .empty .empty) .empty) (.node 9613 .empty .empty)) (.node 10099 (.node 10039 (.node 9781 .empty .empty) .empty) (.node 10333 .empty .empty)))) (.node 15583 (.node 11689 (.node 11353 (.node 10771 (.node 10687 .empty .empty) .empty) (.node 11551 .empty .empty)) (.node 12637 (.node 12487 (.node 12409 .empty .empty) .empty) (.node 15259 .empty .empty))) (.node 19867 (.node 17341 (.node 16963 (.node 15937 .empty .empty) .empty) (.node 19717 .empty .empty)) (.node 20929 (.node 20161 (.node 19993 .empty .empty) .empty) (.node 21187 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4513_ordered : SortedNatTree.ordered row4513 = true := by decide
set_option maxRecDepth 4096 in
theorem row4513_values : SortedNatTree.values row4513 = Data.row4513 := rfl
theorem row4513_checked : SortedNatTree.contains row4513 = Data.row4513.contains := by
  rw [SortedNatTree.contains_eq_list row4513 row4513_ordered, row4513_values]
def row4517 : SortedNatTree.Tree := (.node 11027 (.node 6959 (.node 5849 (.node 5279 (.node 5087 .empty .empty) (.node 5387 .empty .empty)) (.node 6689 (.node 6581 .empty .empty) (.node 6869 .empty .empty))) (.node 9377 (.node 8087 (.node 7523 .empty .empty) (.node 8387 .empty .empty)) (.node 10799 (.node 9689 .empty .empty) .empty))) (.node 15359 (.node 13109 (.node 12473 (.node 12227 .empty .empty) (.node 12941 .empty .empty)) (.node 13763 (.node 13469 .empty .empty) (.node 14657 .empty .empty))) (.node 20147 (.node 17729 (.node 16217 .empty .empty) (.node 19739 .empty .empty)) (.node 21503 (.node 20663 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4517_ordered : SortedNatTree.ordered row4517 = true := by decide
set_option maxRecDepth 4096 in
theorem row4517_values : SortedNatTree.values row4517 = Data.row4517 := rfl
theorem row4517_checked : SortedNatTree.contains row4517 = Data.row4517.contains := by
  rw [SortedNatTree.contains_eq_list row4517 row4517_ordered, row4517_values]
def row4519 : SortedNatTree.Tree := (.node 12589 (.node 8821 (.node 5701 (.node 5209 (.node 4729 (.node 4657 .empty .empty) .empty) (.node 5419 .empty .empty)) (.node 6133 (.node 6043 .empty .empty) (.node 8737 .empty .empty))) (.node 9907 (.node 9199 (.node 9103 (.node 9013 .empty .empty) .empty) (.node 9397 .empty .empty)) (.node 10453 (.node 10321 .empty .empty) (.node 12457 .empty .empty)))) (.node 17497 (.node 15649 (.node 14737 (.node 13921 (.node 13093 .empty .empty) .empty) (.node 15319 .empty .empty)) (.node 16567 (.node 15733 .empty .empty) (.node 16651 .empty .empty))) (.node 20107 (.node 19141 (.node 18679 (.node 18289 .empty .empty) .empty) (.node 19183 .empty .empty)) (.node 20857 (.node 20743 .empty .empty) (.node 21247 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4519_ordered : SortedNatTree.ordered row4519 = true := by decide
set_option maxRecDepth 4096 in
theorem row4519_values : SortedNatTree.values row4519 = Data.row4519 := rfl
theorem row4519_checked : SortedNatTree.contains row4519 = Data.row4519.contains := by
  rw [SortedNatTree.contains_eq_list row4519 row4519_ordered, row4519_values]
def row4523 : SortedNatTree.Tree := (.node 12569 (.node 6911 (.node 5639 (.node 5039 (.node 4967 (.node 4817 .empty .empty) .empty) (.node 5153 (.node 5081 .empty .empty) .empty)) (.node 6551 (.node 6329 (.node 6173 .empty .empty) .empty) (.node 6761 .empty .empty))) (.node 9221 (.node 7937 (.node 7583 (.node 7457 .empty .empty) .empty) (.node 8093 .empty .empty)) (.node 10091 (.node 9743 (.node 9461 .empty .empty) .empty) (.node 10457 .empty .empty)))) (.node 17093 (.node 14249 (.node 13709 (.node 12713 (.node 12611 .empty .empty) .empty) (.node 14081 .empty .empty)) (.node 14867 (.node 14843 (.node 14717 .empty .empty) .empty) (.node 15077 .empty .empty))) (.node 18701 (.node 18353 (.node 17891 (.node 17573 .empty .empty) .empty) (.node 18593 .empty .empty)) (.node 19739 (.node 19433 (.node 19037 .empty .empty) .empty) (.node 21179 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4523_ordered : SortedNatTree.ordered row4523 = true := by decide
set_option maxRecDepth 4096 in
theorem row4523_values : SortedNatTree.values row4523 = Data.row4523 := rfl
theorem row4523_checked : SortedNatTree.contains row4523 = Data.row4523.contains := by
  rw [SortedNatTree.contains_eq_list row4523 row4523_ordered, row4523_values]
def row4547 : SortedNatTree.Tree := (.node 11177 (.node 7283 (.node 6521 (.node 4973 (.node 4943 .empty .empty) (.node 6311 .empty .empty)) (.node 6791 (.node 6689 .empty .empty) (.node 6863 .empty .empty))) (.node 8831 (.node 7919 (.node 7547 .empty .empty) (.node 8597 .empty .empty)) (.node 9851 (.node 8969 .empty .empty) (.node 10889 .empty .empty)))) (.node 16319 (.node 15227 (.node 12101 (.node 11321 .empty .empty) (.node 12149 .empty .empty)) (.node 16103 (.node 15299 .empty .empty) (.node 16139 .empty .empty))) (.node 18287 (.node 17033 (.node 16763 .empty .empty) (.node 17609 .empty .empty)) (.node 19073 (.node 18587 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4547_ordered : SortedNatTree.ordered row4547 = true := by decide
set_option maxRecDepth 4096 in
theorem row4547_values : SortedNatTree.values row4547 = Data.row4547 := rfl
theorem row4547_checked : SortedNatTree.contains row4547 = Data.row4547.contains := by
  rw [SortedNatTree.contains_eq_list row4547 row4547_ordered, row4547_values]
def row4549 : SortedNatTree.Tree := (.node 12109 (.node 7717 (.node 6151 (.node 5851 (.node 5839 (.node 5563 .empty .empty) .empty) (.node 6133 .empty .empty)) (.node 6361 (.node 6343 .empty .empty) (.node 6571 .empty .empty))) (.node 9631 (.node 9127 (.node 8287 .empty .empty) (.node 9133 .empty .empty)) (.node 11161 (.node 10273 .empty .empty) (.node 11239 .empty .empty)))) (.node 16381 (.node 15277 (.node 14011 (.node 12967 (.node 12547 .empty .empty) .empty) (.node 14461 .empty .empty)) (.node 15859 (.node 15667 .empty .empty) (.node 16339 .empty .empty))) (.node 18301 (.node 17401 (.node 17041 .empty .empty) (.node 17707 .empty .empty)) (.node 19681 (.node 19417 .empty .empty) (.node 19753 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4549_ordered : SortedNatTree.ordered row4549 = true := by decide
set_option maxRecDepth 4096 in
theorem row4549_values : SortedNatTree.values row4549 = Data.row4549 := rfl
theorem row4549_checked : SortedNatTree.contains row4549 = Data.row4549.contains := by
  rw [SortedNatTree.contains_eq_list row4549 row4549_ordered, row4549_values]
def row4561 : SortedNatTree.Tree := (.node 10477 (.node 8101 (.node 6073 (.node 5419 (.node 4909 (.node 4783 .empty .empty) .empty) (.node 5527 .empty .empty)) (.node 7723 (.node 6379 .empty .empty) (.node 8017 .empty .empty))) (.node 9283 (.node 8779 (.node 8737 (.node 8389 .empty .empty) .empty) (.node 8863 .empty .empty)) (.node 9907 (.node 9811 .empty .empty) (.node 10321 .empty .empty)))) (.node 13729 (.node 12967 (.node 11833 (.node 11527 (.node 10957 .empty .empty) .empty) (.node 12037 .empty .empty)) (.node 13309 (.node 13249 .empty .empty) (.node 13381 .empty .empty))) (.node 17491 (.node 15271 (.node 14107 .empty .empty) (.node 15817 .empty .empty)) (.node 21031 (.node 19429 .empty .empty) (.node 21319 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4561_ordered : SortedNatTree.ordered row4561 = true := by decide
set_option maxRecDepth 4096 in
theorem row4561_values : SortedNatTree.values row4561 = Data.row4561 := rfl
theorem row4561_checked : SortedNatTree.contains row4561 = Data.row4561.contains := by
  rw [SortedNatTree.contains_eq_list row4561 row4561_ordered, row4561_values]
def row4567 : SortedNatTree.Tree := (.node 15073 (.node 9091 (.node 7789 (.node 7039 (.node 5557 (.node 5323 .empty .empty) .empty) (.node 7417 .empty .empty)) (.node 8467 (.node 7993 .empty .empty) (.node 8719 .empty .empty))) (.node 12829 (.node 10399 (.node 9643 .empty .empty) (.node 10861 .empty .empty)) (.node 14737 (.node 14437 .empty .empty) (.node 14779 .empty .empty)))) (.node 18433 (.node 16369 (.node 15859 (.node 15823 .empty .empty) (.node 16189 .empty .empty)) (.node 17029 (.node 16417 .empty .empty) (.node 17467 .empty .empty))) (.node 19429 (.node 19141 (.node 18517 .empty .empty) (.node 19381 .empty .empty)) (.node 20533 (.node 19507 .empty .empty) (.node 20809 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4567_ordered : SortedNatTree.ordered row4567 = true := by decide
set_option maxRecDepth 4096 in
theorem row4567_values : SortedNatTree.values row4567 = Data.row4567 := rfl
theorem row4567_checked : SortedNatTree.contains row4567 = Data.row4567.contains := by
  rw [SortedNatTree.contains_eq_list row4567 row4567_ordered, row4567_values]
def row4583 : SortedNatTree.Tree := (.node 15773 (.node 9137 (.node 5039 (.node 4871 (.node 4787 (.node 4637 .empty .empty) .empty) (.node 5003 .empty .empty)) (.node 7577 (.node 7523 (.node 5651 .empty .empty) .empty) (.node 8627 .empty .empty))) (.node 11093 (.node 10613 (.node 10607 (.node 9551 .empty .empty) .empty) (.node 10691 .empty .empty)) (.node 13463 (.node 13007 .empty .empty) (.node 14009 .empty .empty)))) (.node 18947 (.node 17729 (.node 17117 (.node 16871 (.node 15959 .empty .empty) .empty) (.node 17477 .empty .empty)) (.node 18713 (.node 18503 .empty .empty) (.node 18899 .empty .empty))) (.node 20249 (.node 19739 (.node 19571 (.node 19181 .empty .empty) .empty) (.node 20231 .empty .empty)) (.node 20939 (.node 20873 .empty .empty) (.node 21317 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4583_ordered : SortedNatTree.ordered row4583 = true := by decide
set_option maxRecDepth 4096 in
theorem row4583_values : SortedNatTree.values row4583 = Data.row4583 := rfl
theorem row4583_checked : SortedNatTree.contains row4583 = Data.row4583.contains := by
  rw [SortedNatTree.contains_eq_list row4583 row4583_ordered, row4583_values]
def row4591 : SortedNatTree.Tree := (.node 11239 (.node 8287 (.node 7213 (.node 5197 (.node 5101 (.node 4651 .empty .empty) .empty) (.node 6121 .empty .empty)) (.node 7621 (.node 7369 .empty .empty) (.node 8167 .empty .empty))) (.node 9883 (.node 9001 (.node 8329 (.node 8293 .empty .empty) .empty) (.node 9007 .empty .empty)) (.node 10627 (.node 9931 .empty .empty) (.node 10903 .empty .empty)))) (.node 17443 (.node 12739 (.node 12109 (.node 11833 (.node 11353 .empty .empty) .empty) (.node 12487 .empty .empty)) (.node 14797 (.node 14011 .empty .empty) (.node 17359 .empty .empty))) (.node 18367 (.node 17707 (.node 17683 .empty .empty) (.node 17851 .empty .empty)) (.node 20443 (.node 18493 .empty .empty) (.node 20611 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4591_ordered : SortedNatTree.ordered row4591 = true := by decide
set_option maxRecDepth 4096 in
theorem row4591_values : SortedNatTree.values row4591 = Data.row4591 := rfl
theorem row4591_checked : SortedNatTree.contains row4591 = Data.row4591.contains := by
  rw [SortedNatTree.contains_eq_list row4591 row4591_ordered, row4591_values]
def row4597 : SortedNatTree.Tree := (.node 9931 (.node 8011 (.node 5749 (.node 5563 (.node 5113 (.node 4969 .empty .empty) .empty) (.node 5581 .empty .empty)) (.node 7213 (.node 6079 (.node 6043 .empty .empty) .empty) (.node 7993 .empty .empty))) (.node 9151 (.node 8389 (.node 8293 (.node 8263 .empty .empty) .empty) (.node 8923 .empty .empty)) (.node 9403 (.node 9397 .empty .empty) (.node 9907 .empty .empty)))) (.node 15661 (.node 14071 (.node 11779 (.node 11437 (.node 11383 .empty .empty) .empty) (.node 12073 .empty .empty)) (.node 15259 (.node 14197 .empty .empty) (.node 15331 .empty .empty))) (.node 19417 (.node 17431 (.node 16987 (.node 16189 .empty .empty) .empty) (.node 18307 .empty .empty)) (.node 20431 (.node 20089 .empty .empty) (.node 21139 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4597_ordered : SortedNatTree.ordered row4597 = true := by decide
set_option maxRecDepth 4096 in
theorem row4597_values : SortedNatTree.values row4597 = Data.row4597 := rfl
theorem row4597_checked : SortedNatTree.contains row4597 = Data.row4597.contains := by
  rw [SortedNatTree.contains_eq_list row4597 row4597_ordered, row4597_values]
def row4603 : SortedNatTree.Tree := (.node 11503 (.node 8599 (.node 6421 (.node 6073 (.node 5569 .empty .empty) (.node 6379 .empty .empty)) (.node 7603 (.node 6577 .empty .empty) .empty)) (.node 9601 (.node 8761 (.node 8731 .empty .empty) (.node 9241 .empty .empty)) (.node 9901 (.node 9769 .empty .empty) .empty))) (.node 17257 (.node 13291 (.node 12637 (.node 12421 .empty .empty) (.node 12841 .empty .empty)) (.node 16573 (.node 14983 .empty .empty) .empty)) (.node 18583 (.node 17707 (.node 17491 .empty .empty) (.node 18127 .empty .empty)) (.node 20533 (.node 20347 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4603_ordered : SortedNatTree.ordered row4603 = true := by decide
set_option maxRecDepth 4096 in
theorem row4603_values : SortedNatTree.values row4603 = Data.row4603 := rfl
theorem row4603_checked : SortedNatTree.contains row4603 = Data.row4603.contains := by
  rw [SortedNatTree.contains_eq_list row4603 row4603_ordered, row4603_values]
def row4621 : SortedNatTree.Tree := (.node 14197 (.node 8287 (.node 6673 (.node 5641 (.node 5347 .empty .empty) (.node 5779 .empty .empty)) (.node 7489 (.node 6679 .empty .empty) (.node 7573 .empty .empty))) (.node 11161 (.node 9403 (.node 9337 .empty .empty) (.node 10651 .empty .empty)) (.node 12421 (.node 11257 .empty .empty) .empty))) (.node 18661 (.node 16033 (.node 15733 (.node 15583 .empty .empty) (.node 15889 .empty .empty)) (.node 17419 (.node 17377 .empty .empty) (.node 17989 .empty .empty))) (.node 20359 (.node 20089 (.node 19531 .empty .empty) (.node 20287 .empty .empty)) (.node 21169 (.node 20773 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4621_ordered : SortedNatTree.ordered row4621 = true := by decide
set_option maxRecDepth 4096 in
theorem row4621_values : SortedNatTree.values row4621 = Data.row4621 := rfl
theorem row4621_checked : SortedNatTree.contains row4621 = Data.row4621.contains := by
  rw [SortedNatTree.contains_eq_list row4621 row4621_ordered, row4621_values]
def row4637 : SortedNatTree.Tree := (.node 12203 (.node 6569 (.node 5471 (.node 5009 (.node 4919 .empty .empty) (.node 5081 .empty .empty)) (.node 6173 (.node 5939 .empty .empty) .empty)) (.node 10253 (.node 7949 (.node 7517 .empty .empty) (.node 9041 .empty .empty)) (.node 11069 (.node 10733 .empty .empty) .empty))) (.node 16217 (.node 14243 (.node 13043 (.node 12983 .empty .empty) (.node 13619 .empty .empty)) (.node 15971 (.node 14759 .empty .empty) .empty)) (.node 18797 (.node 17393 (.node 17351 .empty .empty) (.node 17489 .empty .empty)) (.node 20261 (.node 19661 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4637_ordered : SortedNatTree.ordered row4637 = true := by decide
set_option maxRecDepth 4096 in
theorem row4637_values : SortedNatTree.values row4637 = Data.row4637 := rfl
theorem row4637_checked : SortedNatTree.contains row4637 = Data.row4637.contains := by
  rw [SortedNatTree.contains_eq_list row4637 row4637_ordered, row4637_values]
def row4639 : SortedNatTree.Tree := (.node 14797 (.node 7177 (.node 6121 (.node 5749 (.node 4951 .empty .empty) (.node 5827 .empty .empty)) (.node 6373 (.node 6211 .empty .empty) (.node 6709 .empty .empty))) (.node 9679 (.node 8563 (.node 7237 .empty .empty) (.node 9277 .empty .empty)) (.node 11593 (.node 9973 .empty .empty) (.node 13999 .empty .empty)))) (.node 18919 (.node 17257 (.node 16183 (.node 15679 .empty .empty) (.node 16477 .empty .empty)) (.node 17551 (.node 17431 .empty .empty) (.node 17599 .empty .empty))) (.node 19861 (.node 19483 (.node 19381 .empty .empty) (.node 19501 .empty .empty)) (.node 20113 (.node 20029 .empty .empty) (.node 20161 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4639_ordered : SortedNatTree.ordered row4639 = true := by decide
set_option maxRecDepth 4096 in
theorem row4639_values : SortedNatTree.values row4639 = Data.row4639 := rfl
theorem row4639_checked : SortedNatTree.contains row4639 = Data.row4639.contains := by
  rw [SortedNatTree.contains_eq_list row4639 row4639_ordered, row4639_values]
def row4643 : SortedNatTree.Tree := (.node 10949 (.node 6263 (.node 5471 (.node 5231 (.node 5153 (.node 4877 .empty .empty) .empty) (.node 5237 .empty .empty)) (.node 5717 (.node 5693 .empty .empty) (.node 5849 .empty .empty))) (.node 9161 (.node 6827 (.node 6473 .empty .empty) (.node 7757 .empty .empty)) (.node 10211 (.node 9203 .empty .empty) (.node 10799 .empty .empty)))) (.node 17837 (.node 13049 (.node 11909 (.node 11699 (.node 11579 .empty .empty) .empty) (.node 13007 .empty .empty)) (.node 15083 (.node 14303 .empty .empty) (.node 16823 .empty .empty))) (.node 18947 (.node 18401 (.node 17939 .empty .empty) (.node 18461 .empty .empty)) (.node 20399 (.node 20183 .empty .empty) (.node 20627 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4643_ordered : SortedNatTree.ordered row4643 = true := by decide
set_option maxRecDepth 4096 in
theorem row4643_values : SortedNatTree.values row4643 = Data.row4643 := rfl
theorem row4643_checked : SortedNatTree.contains row4643 = Data.row4643.contains := by
  rw [SortedNatTree.contains_eq_list row4643 row4643_ordered, row4643_values]
def row4649 : SortedNatTree.Tree := (.node 14207 (.node 9461 (.node 7727 (.node 6323 (.node 5987 (.node 5399 .empty .empty) .empty) (.node 6719 .empty .empty)) (.node 8537 (.node 8117 (.node 7853 .empty .empty) .empty) (.node 9221 .empty .empty))) (.node 12641 (.node 12113 (.node 10163 (.node 9857 .empty .empty) .empty) (.node 12251 .empty .empty)) (.node 12713 (.node 12671 .empty .empty) (.node 12893 .empty .empty)))) (.node 18119 (.node 15329 (.node 14951 (.node 14633 (.node 14627 .empty .empty) .empty) (.node 15149 .empty .empty)) (.node 15683 (.node 15497 .empty .empty) (.node 18059 .empty .empty))) (.node 20147 (.node 19583 (.node 19421 (.node 18719 .empty .empty) .empty) (.node 20129 .empty .empty)) (.node 20297 (.node 20219 .empty .empty) (.node 20411 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4649_ordered : SortedNatTree.ordered row4649 = true := by decide
set_option maxRecDepth 4096 in
theorem row4649_values : SortedNatTree.values row4649 = Data.row4649 := rfl
theorem row4649_checked : SortedNatTree.contains row4649 = Data.row4649.contains := by
  rw [SortedNatTree.contains_eq_list row4649 row4649_ordered, row4649_values]
def row4651 : SortedNatTree.Tree := (.node 14347 (.node 9349 (.node 7243 (.node 6991 (.node 4723 .empty .empty) .empty) (.node 9091 (.node 8293 .empty .empty) .empty)) (.node 11527 (.node 10663 (.node 9631 .empty .empty) .empty) (.node 13921 (.node 12763 .empty .empty) .empty))) (.node 18541 (.node 16063 (.node 15643 (.node 14437 .empty .empty) .empty) (.node 17761 (.node 17209 .empty .empty) .empty)) (.node 20521 (.node 20431 (.node 20023 .empty .empty) .empty) (.node 20611 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row4651_ordered : SortedNatTree.ordered row4651 = true := by decide
set_option maxRecDepth 4096 in
theorem row4651_values : SortedNatTree.values row4651 = Data.row4651 := rfl
theorem row4651_checked : SortedNatTree.contains row4651 = Data.row4651.contains := by
  rw [SortedNatTree.contains_eq_list row4651 row4651_ordered, row4651_values]
def row4657 : SortedNatTree.Tree := (.node 11317 (.node 8209 (.node 7177 (.node 5227 (.node 4789 .empty .empty) (.node 6217 .empty .empty)) (.node 7753 (.node 7243 .empty .empty) (.node 7933 .empty .empty))) (.node 9283 (.node 8923 (.node 8737 .empty .empty) (.node 8971 .empty .empty)) (.node 10093 (.node 9433 .empty .empty) (.node 10837 .empty .empty)))) (.node 16057 (.node 14407 (.node 13159 (.node 12853 .empty .empty) (.node 13903 .empty .empty)) (.node 14929 (.node 14797 .empty .empty) (.node 15307 .empty .empty))) (.node 19717 (.node 17449 (.node 16963 .empty .empty) (.node 18013 .empty .empty)) (.node 20173 (.node 19867 .empty .empty) (.node 20743 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4657_ordered : SortedNatTree.ordered row4657 = true := by decide
set_option maxRecDepth 4096 in
theorem row4657_values : SortedNatTree.values row4657 = Data.row4657 := rfl
theorem row4657_checked : SortedNatTree.contains row4657 = Data.row4657.contains := by
  rw [SortedNatTree.contains_eq_list row4657 row4657_ordered, row4657_values]
def row4663 : SortedNatTree.Tree := (.node 12799 (.node 8017 (.node 6637 (.node 5839 (.node 5011 (.node 4993 .empty .empty) .empty) (.node 6619 .empty .empty)) (.node 7753 (.node 7603 .empty .empty) (.node 7879 .empty .empty))) (.node 9811 (.node 9601 (.node 9199 (.node 8233 .empty .empty) .empty) (.node 9787 .empty .empty)) (.node 10399 (.node 9949 .empty .empty) (.node 11113 .empty .empty)))) (.node 17749 (.node 16057 (.node 14923 (.node 14731 (.node 14533 .empty .empty) .empty) (.node 15061 .empty .empty)) (.node 17107 (.node 17077 .empty .empty) (.node 17569 .empty .empty))) (.node 19543 (.node 18367 (.node 17851 .empty .empty) (.node 18439 .empty .empty)) (.node 20479 (.node 19753 .empty .empty) (.node 20773 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4663_ordered : SortedNatTree.ordered row4663 = true := by decide
set_option maxRecDepth 4096 in
theorem row4663_values : SortedNatTree.values row4663 = Data.row4663 := rfl
theorem row4663_checked : SortedNatTree.contains row4663 = Data.row4663.contains := by
  rw [SortedNatTree.contains_eq_list row4663 row4663_ordered, row4663_values]
def row4673 : SortedNatTree.Tree := (.node 10601 (.node 6947 (.node 6359 (.node 5309 (.node 4799 .empty .empty) (.node 5519 .empty .empty)) (.node 6779 (.node 6737 .empty .empty) .empty)) (.node 9551 (.node 8009 (.node 7451 .empty .empty) (.node 9437 .empty .empty)) (.node 10391 (.node 10313 .empty .empty) .empty))) (.node 15461 (.node 14723 (.node 13883 (.node 10937 .empty .empty) (.node 13967 .empty .empty)) (.node 15137 (.node 14957 .empty .empty) .empty)) (.node 20393 (.node 16901 (.node 16481 .empty .empty) (.node 19913 .empty .empty)) (.node 21059 (.node 20639 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4673_ordered : SortedNatTree.ordered row4673 = true := by decide
set_option maxRecDepth 4096 in
theorem row4673_values : SortedNatTree.values row4673 = Data.row4673 := rfl
theorem row4673_checked : SortedNatTree.contains row4673 = Data.row4673.contains := by
  rw [SortedNatTree.contains_eq_list row4673 row4673_ordered, row4673_values]
def row4679 : SortedNatTree.Tree := (.node 10691 (.node 6653 (.node 5717 (.node 5003 (.node 4889 .empty .empty) (.node 5393 .empty .empty)) (.node 6521 (.node 5927 .empty .empty) (.node 6569 .empty .empty))) (.node 9887 (.node 8819 (.node 8783 .empty .empty) (.node 8867 .empty .empty)) (.node 10343 (.node 10337 .empty .empty) .empty))) (.node 16061 (.node 14627 (.node 11597 (.node 11177 .empty .empty) (.node 14033 .empty .empty)) (.node 15131 (.node 14633 .empty .empty) (.node 15647 .empty .empty))) (.node 18413 (.node 16607 (.node 16091 .empty .empty) (.node 18401 .empty .empty)) (.node 20879 (.node 20183 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4679_ordered : SortedNatTree.ordered row4679 = true := by decide
set_option maxRecDepth 4096 in
theorem row4679_values : SortedNatTree.values row4679 = Data.row4679 := rfl
theorem row4679_checked : SortedNatTree.contains row4679 = Data.row4679.contains := by
  rw [SortedNatTree.contains_eq_list row4679 row4679_ordered, row4679_values]
def row4691 : SortedNatTree.Tree := (.node 16001 (.node 13457 (.node 6581 (.node 5987 (.node 4817 .empty .empty) (.node 6521 .empty .empty)) (.node 7457 (.node 6911 .empty .empty) (.node 12239 .empty .empty))) (.node 14699 (.node 14249 (.node 13709 .empty .empty) (.node 14519 .empty .empty)) (.node 15887 (.node 15359 .empty .empty) .empty))) (.node 18311 (.node 17471 (.node 16481 (.node 16127 .empty .empty) (.node 17027 .empty .empty)) (.node 17921 (.node 17609 .empty .empty) .empty)) (.node 19571 (.node 19163 (.node 18719 .empty .empty) (.node 19181 .empty .empty)) (.node 20369 (.node 19841 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4691_ordered : SortedNatTree.ordered row4691 = true := by decide
set_option maxRecDepth 4096 in
theorem row4691_values : SortedNatTree.values row4691 = Data.row4691 := rfl
theorem row4691_checked : SortedNatTree.contains row4691 = Data.row4691.contains := by
  rw [SortedNatTree.contains_eq_list row4691 row4691_ordered, row4691_values]
def row4703 : SortedNatTree.Tree := (.node 11057 (.node 6653 (.node 6221 (.node 5657 (.node 5531 .empty .empty) (.node 6131 .empty .empty)) (.node 6329 (.node 6257 .empty .empty) (.node 6359 .empty .empty))) (.node 8837 (.node 8543 (.node 8081 .empty .empty) (.node 8627 .empty .empty)) (.node 9743 (.node 9371 .empty .empty) (.node 9749 .empty .empty)))) (.node 17483 (.node 13751 (.node 13331 (.node 11483 .empty .empty) (.node 13397 .empty .empty)) (.node 16127 (.node 15641 .empty .empty) (.node 16607 .empty .empty))) (.node 19991 (.node 18869 (.node 17609 .empty .empty) (.node 19697 .empty .empty)) (.node 21011 (.node 20051 .empty .empty) (.node 21017 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4703_ordered : SortedNatTree.ordered row4703 = true := by decide
set_option maxRecDepth 4096 in
theorem row4703_values : SortedNatTree.values row4703 = Data.row4703 := rfl
theorem row4703_checked : SortedNatTree.contains row4703 = Data.row4703.contains := by
  rw [SortedNatTree.contains_eq_list row4703 row4703_ordered, row4703_values]
def row4721 : SortedNatTree.Tree := (.node 10883 (.node 7949 (.node 5501 (.node 4919 (.node 4877 .empty .empty) (.node 5417 .empty .empty)) (.node 6971 (.node 5813 .empty .empty) .empty)) (.node 9257 (.node 9209 (.node 8753 .empty .empty) .empty) (.node 10589 (.node 9803 .empty .empty) .empty))) (.node 17483 (.node 14843 (.node 13751 (.node 12269 .empty .empty) (.node 14627 .empty .empty)) (.node 15731 (.node 14891 .empty .empty) .empty)) (.node 19997 (.node 19031 (.node 18287 .empty .empty) .empty) (.node 20921 (.node 20507 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4721_ordered : SortedNatTree.ordered row4721 = true := by decide
set_option maxRecDepth 4096 in
theorem row4721_values : SortedNatTree.values row4721 = Data.row4721 := rfl
theorem row4721_checked : SortedNatTree.contains row4721 = Data.row4721.contains := by
  rw [SortedNatTree.contains_eq_list row4721 row4721_ordered, row4721_values]
def row4723 : SortedNatTree.Tree := (.node 12979 (.node 8221 (.node 6361 (.node 5623 (.node 5521 .empty .empty) (.node 5869 .empty .empty)) (.node 7369 (.node 7243 .empty .empty) .empty)) (.node 9721 (.node 8923 (.node 8839 .empty .empty) (.node 9613 .empty .empty)) (.node 10159 (.node 9901 .empty .empty) .empty))) (.node 17359 (.node 15919 (.node 13249 (.node 13099 .empty .empty) (.node 15427 .empty .empty)) (.node 16561 (.node 16333 .empty .empty) .empty)) (.node 19699 (.node 18661 (.node 18439 .empty .empty) (.node 19207 .empty .empty)) (.node 21139 (.node 20389 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4723_ordered : SortedNatTree.ordered row4723 = true := by decide
set_option maxRecDepth 4096 in
theorem row4723_values : SortedNatTree.values row4723 = Data.row4723 := rfl
theorem row4723_checked : SortedNatTree.contains row4723 = Data.row4723.contains := by
  rw [SortedNatTree.contains_eq_list row4723 row4723_ordered, row4723_values]
def row4729 : SortedNatTree.Tree := (.node 12619 (.node 8863 (.node 5791 (.node 5077 (.node 4999 (.node 4909 .empty .empty) .empty) (.node 5749 (.node 5407 .empty .empty) .empty)) (.node 7603 (.node 7207 (.node 6961 .empty .empty) .empty) (.node 8431 (.node 7963 .empty .empty) .empty))) (.node 11239 (.node 9601 (.node 9181 (.node 8887 .empty .empty) .empty) (.node 10831 (.node 9859 .empty .empty) .empty)) (.node 12301 (.node 12289 (.node 11251 .empty .empty) .empty) (.node 12457 .empty .empty)))) (.node 18289 (.node 15559 (.node 15277 (.node 14437 (.node 12829 .empty .empty) .empty) (.node 15439 (.node 15349 .empty .empty) .empty)) (.node 16921 (.node 16879 (.node 16651 .empty .empty) .empty) (.node 18181 .empty .empty))) (.node 19927 (.node 19441 (.node 19219 (.node 18637 .empty .empty) .empty) (.node 19597 (.node 19483 .empty .empty) .empty)) (.node 20743 (.node 20269 (.node 20143 .empty .empty) .empty) (.node 20947 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4729_ordered : SortedNatTree.ordered row4729 = true := by decide
set_option maxRecDepth 4096 in
theorem row4729_values : SortedNatTree.values row4729 = Data.row4729 := rfl
theorem row4729_checked : SortedNatTree.contains row4729 = Data.row4729.contains := by
  rw [SortedNatTree.contains_eq_list row4729 row4729_ordered, row4729_values]
def row4733 : SortedNatTree.Tree := (.node 11903 (.node 7793 (.node 7079 (.node 6689 (.node 6269 (.node 5237 .empty .empty) .empty) (.node 6911 .empty .empty)) (.node 7331 (.node 7211 .empty .empty) (.node 7517 .empty .empty))) (.node 9437 (.node 8849 (.node 7919 (.node 7907 .empty .empty) .empty) (.node 9029 .empty .empty)) (.node 10259 (.node 10037 .empty .empty) (.node 10289 .empty .empty)))) (.node 16001 (.node 14207 (.node 13187 (.node 12923 (.node 12821 .empty .empty) .empty) (.node 13469 .empty .empty)) (.node 15527 (.node 15329 .empty .empty) (.node 15959 .empty .empty))) (.node 19433 (.node 18413 (.node 17909 (.node 16223 .empty .empty) .empty) (.node 18617 .empty .empty)) (.node 19961 (.node 19853 .empty .empty) (.node 20873 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4733_ordered : SortedNatTree.ordered row4733 = true := by decide
set_option maxRecDepth 4096 in
theorem row4733_values : SortedNatTree.values row4733 = Data.row4733 := rfl
theorem row4733_checked : SortedNatTree.contains row4733 = Data.row4733.contains := by
  rw [SortedNatTree.contains_eq_list row4733 row4733_ordered, row4733_values]
def row4751 : SortedNatTree.Tree := (.node 10631 (.node 7331 (.node 5879 (.node 5297 (.node 5099 .empty .empty) (.node 5711 .empty .empty)) (.node 6317 (.node 6113 .empty .empty) (.node 6659 .empty .empty))) (.node 8837 (.node 7649 (.node 7607 .empty .empty) (.node 8609 .empty .empty)) (.node 9929 (.node 8963 .empty .empty) .empty))) (.node 15737 (.node 13901 (.node 13217 (.node 13007 .empty .empty) (.node 13763 .empty .empty)) (.node 14951 (.node 14033 .empty .empty) (.node 15581 .empty .empty))) (.node 18257 (.node 17789 (.node 17519 .empty .empty) (.node 18233 .empty .empty)) (.node 20939 (.node 18593 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4751_ordered : SortedNatTree.ordered row4751 = true := by decide
set_option maxRecDepth 4096 in
theorem row4751_values : SortedNatTree.values row4751 = Data.row4751 := rfl
theorem row4751_checked : SortedNatTree.contains row4751 = Data.row4751.contains := by
  rw [SortedNatTree.contains_eq_list row4751 row4751_ordered, row4751_values]
def row4759 : SortedNatTree.Tree := (.node 14479 (.node 9661 (.node 6151 (.node 5227 (.node 4903 .empty .empty) (.node 5701 .empty .empty)) (.node 9511 (.node 6823 .empty .empty) .empty)) (.node 12097 (.node 11827 (.node 11551 .empty .empty) .empty) (.node 13597 (.node 12919 .empty .empty) .empty))) (.node 19237 (.node 17191 (.node 16087 (.node 14797 .empty .empty) (.node 16729 .empty .empty)) (.node 18787 (.node 17539 .empty .empty) .empty)) (.node 20983 (.node 20047 (.node 19333 .empty .empty) .empty) (.node 21211 (.node 21169 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row4759_ordered : SortedNatTree.ordered row4759 = true := by decide
set_option maxRecDepth 4096 in
theorem row4759_values : SortedNatTree.values row4759 = Data.row4759 := rfl
theorem row4759_checked : SortedNatTree.contains row4759 = Data.row4759.contains := by
  rw [SortedNatTree.contains_eq_list row4759 row4759_ordered, row4759_values]
def row4783 : SortedNatTree.Tree := (.node 13963 (.node 7879 (.node 6073 (.node 5167 (.node 4999 .empty .empty) (.node 5779 .empty .empty)) (.node 7219 (.node 6781 .empty .empty) (.node 7873 .empty .empty))) (.node 9613 (.node 9049 (.node 8803 .empty .empty) (.node 9199 .empty .empty)) (.node 10303 (.node 9787 .empty .empty) (.node 11311 .empty .empty)))) (.node 18223 (.node 15601 (.node 14293 (.node 14251 .empty .empty) (.node 15493 .empty .empty)) (.node 17911 (.node 16087 .empty .empty) (.node 17971 .empty .empty))) (.node 18661 (.node 18439 (.node 18397 .empty .empty) (.node 18583 .empty .empty)) (.node 19843 (.node 18793 .empty .empty) (.node 20641 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row4783_ordered : SortedNatTree.ordered row4783 = true := by decide
set_option maxRecDepth 4096 in
theorem row4783_values : SortedNatTree.values row4783 = Data.row4783 := rfl
theorem row4783_checked : SortedNatTree.contains row4783 = Data.row4783.contains := by
  rw [SortedNatTree.contains_eq_list row4783 row4783_ordered, row4783_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
