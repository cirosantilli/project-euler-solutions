import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row8179 : SortedNatTree.Tree := (.node 11833 (.node 10663 (.node 9463 (.node 9181 (.node 8293 .empty .empty) .empty) (.node 10657 .empty .empty)) (.node 11587 (.node 11527 (.node 10831 .empty .empty) .empty) (.node 11821 .empty .empty))) (.node 13339 (.node 12889 (.node 12541 (.node 12037 .empty .empty) .empty) (.node 13003 .empty .empty)) (.node 17167 (.node 16657 (.node 13963 .empty .empty) .empty) (.node 17761 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8179_ordered : SortedNatTree.ordered row8179 = true := by decide
set_option maxRecDepth 4096 in
theorem row8179_values : SortedNatTree.values row8179 = Data.row8179 := rfl
theorem row8179_checked : SortedNatTree.contains row8179 = Data.row8179.contains := by
  rw [SortedNatTree.contains_eq_list row8179 row8179_ordered, row8179_values]
def row8191 : SortedNatTree.Tree := (.node 14983 (.node 11287 (.node 9769 (.node 9547 (.node 8737 .empty .empty) .empty) (.node 10009 .empty .empty)) (.node 14419 (.node 14221 (.node 12043 .empty .empty) .empty) (.node 14713 .empty .empty))) (.node 16447 (.node 16183 (.node 15667 (.node 15313 .empty .empty) .empty) (.node 16381 .empty .empty)) (.node 17077 (.node 16759 (.node 16633 .empty .empty) .empty) (.node 17737 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8191_ordered : SortedNatTree.ordered row8191 = true := by decide
set_option maxRecDepth 4096 in
theorem row8191_values : SortedNatTree.values row8191 = Data.row8191 := rfl
theorem row8191_checked : SortedNatTree.contains row8191 = Data.row8191.contains := by
  rw [SortedNatTree.contains_eq_list row8191 row8191_ordered, row8191_values]
def row8209 : SortedNatTree.Tree := (.node 14293 (.node 10267 (.node 9649 (.node 8941 (.node 8641 .empty .empty) .empty) (.node 9859 .empty .empty)) (.node 12637 (.node 12109 .empty .empty) (.node 12973 .empty .empty))) (.node 17341 (.node 16633 (.node 16249 .empty .empty) (.node 17191 .empty .empty)) (.node 17569 (.node 17491 .empty .empty) (.node 17599 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8209_ordered : SortedNatTree.ordered row8209 = true := by decide
set_option maxRecDepth 4096 in
theorem row8209_values : SortedNatTree.values row8209 = Data.row8209 := rfl
theorem row8209_checked : SortedNatTree.contains row8209 = Data.row8209.contains := by
  rw [SortedNatTree.contains_eq_list row8209 row8209_ordered, row8209_values]
def row8219 : SortedNatTree.Tree := (.node 12569 (.node 11369 (.node 10343 (.node 9239 (.node 8819 .empty .empty) .empty) (.node 11117 .empty .empty)) (.node 12491 (.node 12251 .empty .empty) (.node 12503 .empty .empty))) (.node 15569 (.node 14699 (.node 13397 (.node 13163 .empty .empty) .empty) (.node 15527 .empty .empty)) (.node 16451 (.node 16349 .empty .empty) (.node 17573 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8219_ordered : SortedNatTree.ordered row8219 = true := by decide
set_option maxRecDepth 4096 in
theorem row8219_values : SortedNatTree.values row8219 = Data.row8219 := rfl
theorem row8219_checked : SortedNatTree.contains row8219 = Data.row8219.contains := by
  rw [SortedNatTree.contains_eq_list row8219 row8219_ordered, row8219_values]
def row8221 : SortedNatTree.Tree := (.node 14143 (.node 11443 (.node 9433 (.node 9127 .empty .empty) (.node 9649 .empty .empty)) (.node 13177 (.node 12457 .empty .empty) (.node 13297 .empty .empty))) (.node 17041 (.node 16693 (.node 14221 .empty .empty) (.node 16747 .empty .empty)) (.node 17509 (.node 17077 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8221_ordered : SortedNatTree.ordered row8221 = true := by decide
set_option maxRecDepth 4096 in
theorem row8221_values : SortedNatTree.values row8221 = Data.row8221 := rfl
theorem row8221_checked : SortedNatTree.contains row8221 = Data.row8221.contains := by
  rw [SortedNatTree.contains_eq_list row8221 row8221_ordered, row8221_values]
def row8231 : SortedNatTree.Tree := (.node 11087 (.node 9623 (.node 9203 (.node 8273 .empty .empty) (.node 9467 .empty .empty)) (.node 10487 (.node 9689 .empty .empty) .empty)) (.node 14249 (.node 12041 (.node 11897 .empty .empty) (.node 13037 .empty .empty)) (.node 17291 (.node 16223 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8231_ordered : SortedNatTree.ordered row8231 = true := by decide
set_option maxRecDepth 4096 in
theorem row8231_values : SortedNatTree.values row8231 = Data.row8231 := rfl
theorem row8231_checked : SortedNatTree.contains row8231 = Data.row8231.contains := by
  rw [SortedNatTree.contains_eq_list row8231 row8231_ordered, row8231_values]
def row8233 : SortedNatTree.Tree := (.node 10597 (.node 9067 (.node 8677 (.node 8563 (.node 8329 .empty .empty) .empty) (.node 8863 .empty .empty)) (.node 9769 (.node 9199 .empty .empty) (.node 9949 .empty .empty))) (.node 15289 (.node 13411 (.node 13171 (.node 10651 .empty .empty) .empty) (.node 14533 .empty .empty)) (.node 17389 (.node 15907 .empty .empty) (.node 17791 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8233_ordered : SortedNatTree.ordered row8233 = true := by decide
set_option maxRecDepth 4096 in
theorem row8233_values : SortedNatTree.values row8233 = Data.row8233 := rfl
theorem row8233_checked : SortedNatTree.contains row8233 = Data.row8233.contains := by
  rw [SortedNatTree.contains_eq_list row8233 row8233_ordered, row8233_values]
def row8237 : SortedNatTree.Tree := (.node 12119 (.node 10103 (.node 9851 (.node 9677 (.node 9623 .empty .empty) .empty) (.node 10079 .empty .empty)) (.node 10739 (.node 10337 .empty .empty) (.node 11177 .empty .empty))) (.node 14561 (.node 12959 (.node 12689 (.node 12269 .empty .empty) .empty) (.node 13901 .empty .empty)) (.node 15791 (.node 14897 .empty .empty) (.node 17333 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8237_ordered : SortedNatTree.ordered row8237 = true := by decide
set_option maxRecDepth 4096 in
theorem row8237_values : SortedNatTree.values row8237 = Data.row8237 := rfl
theorem row8237_checked : SortedNatTree.contains row8237 = Data.row8237.contains := by
  rw [SortedNatTree.contains_eq_list row8237 row8237_ordered, row8237_values]
def row8243 : SortedNatTree.Tree := (.node 13751 (.node 9629 (.node 9473 (.node 9209 (.node 8963 .empty .empty) .empty) (.node 9587 .empty .empty)) (.node 11321 (.node 11177 .empty .empty) (.node 12941 .empty .empty))) (.node 14537 (.node 14159 (.node 13907 .empty .empty) (.node 14423 .empty .empty)) (.node 15299 (.node 15083 .empty .empty) (.node 17657 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8243_ordered : SortedNatTree.ordered row8243 = true := by decide
set_option maxRecDepth 4096 in
theorem row8243_values : SortedNatTree.values row8243 = Data.row8243 := rfl
theorem row8243_checked : SortedNatTree.contains row8243 = Data.row8243.contains := by
  rw [SortedNatTree.contains_eq_list row8243 row8243_ordered, row8243_values]
def row8263 : SortedNatTree.Tree := (.node 14173 (.node 13003 (.node 9511 (.node 9433 (.node 9343 .empty .empty) .empty) (.node 9967 (.node 9547 .empty .empty) .empty)) (.node 13729 (.node 13681 (.node 13633 .empty .empty) .empty) (.node 14143 .empty .empty))) (.node 16903 (.node 15313 (.node 14737 (.node 14341 .empty .empty) .empty) (.node 16567 .empty .empty)) (.node 17509 (.node 17377 (.node 17203 .empty .empty) .empty) (.node 17659 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8263_ordered : SortedNatTree.ordered row8263 = true := by decide
set_option maxRecDepth 4096 in
theorem row8263_values : SortedNatTree.values row8263 = Data.row8263 := rfl
theorem row8263_checked : SortedNatTree.contains row8263 = Data.row8263.contains := by
  rw [SortedNatTree.contains_eq_list row8263 row8263_ordered, row8263_values]
def row8269 : SortedNatTree.Tree := (.node 13873 (.node 11383 (.node 8821 (.node 8713 (.node 8527 .empty .empty) .empty) (.node 9787 (.node 9319 .empty .empty) .empty)) (.node 12577 (.node 12049 (.node 11941 .empty .empty) .empty) (.node 12781 (.node 12637 .empty .empty) .empty))) (.node 16231 (.node 14779 (.node 14107 (.node 13903 .empty .empty) .empty) (.node 15217 (.node 15121 .empty .empty) .empty)) (.node 17431 (.node 17239 (.node 16477 .empty .empty) .empty) (.node 17737 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8269_ordered : SortedNatTree.ordered row8269 = true := by decide
set_option maxRecDepth 4096 in
theorem row8269_values : SortedNatTree.values row8269 = Data.row8269 := rfl
theorem row8269_checked : SortedNatTree.contains row8269 = Data.row8269.contains := by
  rw [SortedNatTree.contains_eq_list row8269 row8269_ordered, row8269_values]
def row8273 : SortedNatTree.Tree := (.node 12113 (.node 10079 (.node 9227 (.node 8681 (.node 8543 .empty .empty) .empty) (.node 10007 .empty .empty)) (.node 11717 (.node 11423 .empty .empty) (.node 11831 .empty .empty))) (.node 14771 (.node 14627 (.node 13217 (.node 12347 .empty .empty) .empty) (.node 14759 .empty .empty)) (.node 15767 (.node 15131 .empty .empty) (.node 17657 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8273_ordered : SortedNatTree.ordered row8273 = true := by decide
set_option maxRecDepth 4096 in
theorem row8273_values : SortedNatTree.values row8273 = Data.row8273 := rfl
theorem row8273_checked : SortedNatTree.contains row8273 = Data.row8273.contains := by
  rw [SortedNatTree.contains_eq_list row8273 row8273_ordered, row8273_values]
def row8287 : SortedNatTree.Tree := (.node 12433 (.node 10357 (.node 9721 (.node 9133 .empty .empty) (.node 9859 .empty .empty)) (.node 12253 (.node 11173 .empty .empty) .empty)) (.node 14713 (.node 14347 (.node 13597 .empty .empty) .empty) (.node 17713 (.node 16741 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8287_ordered : SortedNatTree.ordered row8287 = true := by decide
set_option maxRecDepth 4096 in
theorem row8287_values : SortedNatTree.values row8287 = Data.row8287 := rfl
theorem row8287_checked : SortedNatTree.contains row8287 = Data.row8287.contains := by
  rw [SortedNatTree.contains_eq_list row8287 row8287_ordered, row8287_values]
def row8291 : SortedNatTree.Tree := (.node 12071 (.node 10259 (.node 9533 (.node 8747 (.node 8573 .empty .empty) (.node 8861 .empty .empty)) (.node 9887 (.node 9587 .empty .empty) .empty)) (.node 11279 (.node 11057 (.node 10589 .empty .empty) .empty) (.node 11903 (.node 11351 .empty .empty) .empty))) (.node 13877 (.node 12959 (.node 12953 (.node 12161 .empty .empty) .empty) (.node 13829 (.node 13457 .empty .empty) .empty)) (.node 15629 (.node 15473 (.node 14591 .empty .empty) .empty) (.node 16229 (.node 15971 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row8291_ordered : SortedNatTree.ordered row8291 = true := by decide
set_option maxRecDepth 4096 in
theorem row8291_values : SortedNatTree.values row8291 = Data.row8291 := rfl
theorem row8291_checked : SortedNatTree.contains row8291 = Data.row8291.contains := by
  rw [SortedNatTree.contains_eq_list row8291 row8291_ordered, row8291_values]
def row8293 : SortedNatTree.Tree := (.node 11719 (.node 9697 (.node 8689 (.node 8431 (.node 8389 .empty .empty) .empty) (.node 9133 .empty .empty)) (.node 10501 (.node 10039 (.node 9949 .empty .empty) .empty) (.node 11467 .empty .empty))) (.node 15121 (.node 12973 (.node 12433 (.node 12157 .empty .empty) .empty) (.node 13411 .empty .empty)) (.node 15541 (.node 15307 .empty .empty) (.node 17389 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8293_ordered : SortedNatTree.ordered row8293 = true := by decide
set_option maxRecDepth 4096 in
theorem row8293_values : SortedNatTree.values row8293 = Data.row8293 := rfl
theorem row8293_checked : SortedNatTree.contains row8293 = Data.row8293.contains := by
  rw [SortedNatTree.contains_eq_list row8293 row8293_ordered, row8293_values]
def row8297 : SortedNatTree.Tree := (.node 13007 (.node 10631 (.node 9161 (.node 9059 (.node 9041 .empty .empty) .empty) (.node 10457 .empty .empty)) (.node 11423 (.node 10709 .empty .empty) (.node 12899 .empty .empty))) (.node 15773 (.node 15233 (.node 15077 (.node 14387 .empty .empty) .empty) (.node 15329 .empty .empty)) (.node 16451 (.node 16193 .empty .empty) (.node 16661 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8297_ordered : SortedNatTree.ordered row8297 = true := by decide
set_option maxRecDepth 4096 in
theorem row8297_values : SortedNatTree.values row8297 = Data.row8297 := rfl
theorem row8297_checked : SortedNatTree.contains row8297 = Data.row8297.contains := by
  rw [SortedNatTree.contains_eq_list row8297 row8297_ordered, row8297_values]
def row8311 : SortedNatTree.Tree := (.node 14281 (.node 11299 (.node 9349 (.node 8377 (.node 8353 .empty .empty) .empty) (.node 10459 .empty .empty)) (.node 13033 (.node 12973 (.node 12109 .empty .empty) .empty) (.node 13147 .empty .empty))) (.node 15013 (.node 14737 (.node 14653 (.node 14551 .empty .empty) .empty) (.node 14923 .empty .empty)) (.node 15217 (.node 15073 .empty .empty) (.node 17239 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8311_ordered : SortedNatTree.ordered row8311 = true := by decide
set_option maxRecDepth 4096 in
theorem row8311_values : SortedNatTree.values row8311 = Data.row8311 := rfl
theorem row8311_checked : SortedNatTree.contains row8311 = Data.row8311.contains := by
  rw [SortedNatTree.contains_eq_list row8311 row8311_ordered, row8311_values]
def row8317 : SortedNatTree.Tree := (.node 12907 (.node 10357 (.node 9781 (.node 8527 .empty .empty) .empty) (.node 11887 .empty .empty)) (.node 15427 (.node 14731 (.node 13873 .empty .empty) .empty) (.node 16759 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8317_ordered : SortedNatTree.ordered row8317 = true := by decide
set_option maxRecDepth 4096 in
theorem row8317_values : SortedNatTree.values row8317 = Data.row8317 := rfl
theorem row8317_checked : SortedNatTree.contains row8317 = Data.row8317.contains := by
  rw [SortedNatTree.contains_eq_list row8317 row8317_ordered, row8317_values]
def row8329 : SortedNatTree.Tree := (.node 11353 (.node 10303 (.node 8539 (.node 8461 .empty .empty) .empty) (.node 10627 (.node 10333 .empty .empty) .empty)) (.node 14431 (.node 13399 (.node 12037 .empty .empty) .empty) (.node 16033 (.node 15427 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8329_ordered : SortedNatTree.ordered row8329 = true := by decide
set_option maxRecDepth 4096 in
theorem row8329_values : SortedNatTree.values row8329 = Data.row8329 := rfl
theorem row8329_checked : SortedNatTree.contains row8329 = Data.row8329.contains := by
  rw [SortedNatTree.contains_eq_list row8329 row8329_ordered, row8329_values]
def row8353 : SortedNatTree.Tree := (.node 13183 (.node 10639 (.node 8713 (.node 8419 .empty .empty) (.node 9829 .empty .empty)) (.node 11251 (.node 11047 .empty .empty) (.node 11317 .empty .empty))) (.node 15619 (.node 13789 (.node 13711 .empty .empty) (.node 14551 .empty .empty)) (.node 17443 (.node 15643 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8353_ordered : SortedNatTree.ordered row8353 = true := by decide
set_option maxRecDepth 4096 in
theorem row8353_values : SortedNatTree.values row8353 = Data.row8353 := rfl
theorem row8353_checked : SortedNatTree.contains row8353 = Data.row8353.contains := by
  rw [SortedNatTree.contains_eq_list row8353 row8353_ordered, row8353_values]
def row8363 : SortedNatTree.Tree := (.node 13217 (.node 11423 (.node 10067 (.node 9029 (.node 8543 .empty .empty) .empty) (.node 10253 .empty .empty)) (.node 12437 (.node 12143 (.node 12011 .empty .empty) .empty) (.node 12941 .empty .empty))) (.node 16361 (.node 14153 (.node 13781 (.node 13337 .empty .empty) .empty) (.node 15101 .empty .empty)) (.node 16763 (.node 16493 .empty .empty) (.node 17627 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8363_ordered : SortedNatTree.ordered row8363 = true := by decide
set_option maxRecDepth 4096 in
theorem row8363_values : SortedNatTree.values row8363 = Data.row8363 := rfl
theorem row8363_checked : SortedNatTree.contains row8363 = Data.row8363.contains := by
  rw [SortedNatTree.contains_eq_list row8363 row8363_ordered, row8363_values]
def row8369 : SortedNatTree.Tree := (.node 12281 (.node 10631 (.node 9221 (.node 8663 (.node 8627 .empty .empty) .empty) (.node 10259 .empty .empty)) (.node 11897 (.node 11027 .empty .empty) (.node 12119 .empty .empty))) (.node 16349 (.node 15461 (.node 13967 (.node 13241 .empty .empty) .empty) (.node 16193 .empty .empty)) (.node 17099 (.node 17027 .empty .empty) (.node 17489 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8369_ordered : SortedNatTree.ordered row8369 = true := by decide
set_option maxRecDepth 4096 in
theorem row8369_values : SortedNatTree.values row8369 = Data.row8369 := rfl
theorem row8369_checked : SortedNatTree.contains row8369 = Data.row8369.contains := by
  rw [SortedNatTree.contains_eq_list row8369 row8369_ordered, row8369_values]
def row8377 : SortedNatTree.Tree := (.node 11689 (.node 9187 (.node 9001 (.node 8713 .empty .empty) (.node 9103 .empty .empty)) (.node 11131 (.node 10369 .empty .empty) .empty)) (.node 14029 (.node 13399 (.node 12619 .empty .empty) .empty) (.node 17191 (.node 15349 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8377_ordered : SortedNatTree.ordered row8377 = true := by decide
set_option maxRecDepth 4096 in
theorem row8377_values : SortedNatTree.values row8377 = Data.row8377 := rfl
theorem row8377_checked : SortedNatTree.contains row8377 = Data.row8377.contains := by
  rw [SortedNatTree.contains_eq_list row8377 row8377_ordered, row8377_values]
def row8387 : SortedNatTree.Tree := (.node 12473 (.node 10937 (.node 9491 (.node 8849 .empty .empty) .empty) (.node 11717 (.node 11579 .empty .empty) .empty)) (.node 16139 (.node 16127 (.node 13121 .empty .empty) .empty) (.node 17027 (.node 16883 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8387_ordered : SortedNatTree.ordered row8387 = true := by decide
set_option maxRecDepth 4096 in
theorem row8387_values : SortedNatTree.values row8387 = Data.row8387 := rfl
theorem row8387_checked : SortedNatTree.contains row8387 = Data.row8387.contains := by
  rw [SortedNatTree.contains_eq_list row8387 row8387_ordered, row8387_values]
def row8389 : SortedNatTree.Tree := (.node 10939 (.node 9007 (.node 8887 (.node 8677 .empty .empty) (.node 8929 .empty .empty)) (.node 9643 (.node 9133 .empty .empty) (.node 10903 .empty .empty))) (.node 14149 (.node 13219 (.node 12007 .empty .empty) (.node 13687 .empty .empty)) (.node 15619 (.node 15013 .empty .empty) (.node 17623 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8389_ordered : SortedNatTree.ordered row8389 = true := by decide
set_option maxRecDepth 4096 in
theorem row8389_values : SortedNatTree.values row8389 = Data.row8389 := rfl
theorem row8389_checked : SortedNatTree.contains row8389 = Data.row8389.contains := by
  rw [SortedNatTree.contains_eq_list row8389 row8389_ordered, row8389_values]
def row8419 : SortedNatTree.Tree := (.node 9949 (.node 9319 (.node 8923 (.node 8641 .empty .empty) (.node 9277 .empty .empty)) (.node 9883 (.node 9649 .empty .empty) .empty)) (.node 15937 (.node 10789 (.node 10711 .empty .empty) (.node 11503 .empty .empty)) (.node 17443 (.node 17317 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8419_ordered : SortedNatTree.ordered row8419 = true := by decide
set_option maxRecDepth 4096 in
theorem row8419_values : SortedNatTree.values row8419 = Data.row8419 := rfl
theorem row8419_checked : SortedNatTree.contains row8419 = Data.row8419.contains := by
  rw [SortedNatTree.contains_eq_list row8419 row8419_ordered, row8419_values]
def row8423 : SortedNatTree.Tree := (.node 12791 (.node 9137 (.node 8573 (.node 8513 .empty .empty) (.node 8699 .empty .empty)) (.node 11213 (.node 9377 .empty .empty) (.node 12269 .empty .empty))) (.node 13487 (.node 13397 (.node 13151 .empty .empty) (.node 13463 .empty .empty)) (.node 15629 (.node 14639 .empty .empty) (.node 16253 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8423_ordered : SortedNatTree.ordered row8423 = true := by decide
set_option maxRecDepth 4096 in
theorem row8423_values : SortedNatTree.values row8423 = Data.row8423 := rfl
theorem row8423_checked : SortedNatTree.contains row8423 = Data.row8423.contains := by
  rw [SortedNatTree.contains_eq_list row8423 row8423_ordered, row8423_values]
def row8429 : SortedNatTree.Tree := (.node 13457 (.node 10457 (.node 8831 (.node 8807 (.node 8693 .empty .empty) .empty) (.node 10151 .empty .empty)) (.node 12821 (.node 11411 .empty .empty) (.node 13121 .empty .empty))) (.node 15497 (.node 15413 (.node 14543 (.node 14447 .empty .empty) .empty) (.node 15467 .empty .empty)) (.node 16901 (.node 15959 .empty .empty) (.node 16979 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8429_ordered : SortedNatTree.ordered row8429 = true := by decide
set_option maxRecDepth 4096 in
theorem row8429_values : SortedNatTree.values row8429 = Data.row8429 := rfl
theorem row8429_checked : SortedNatTree.contains row8429 = Data.row8429.contains := by
  rw [SortedNatTree.contains_eq_list row8429 row8429_ordered, row8429_values]
def row8431 : SortedNatTree.Tree := (.node 12583 (.node 10369 (.node 9049 (.node 9013 .empty .empty) (.node 9649 .empty .empty)) (.node 12421 (.node 11251 .empty .empty) (.node 12511 .empty .empty))) (.node 16111 (.node 15193 (.node 13309 .empty .empty) (.node 15913 .empty .empty)) (.node 17293 (.node 16987 .empty .empty) (.node 17581 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8431_ordered : SortedNatTree.ordered row8431 = true := by decide
set_option maxRecDepth 4096 in
theorem row8431_values : SortedNatTree.values row8431 = Data.row8431 := rfl
theorem row8431_checked : SortedNatTree.contains row8431 = Data.row8431.contains := by
  rw [SortedNatTree.contains_eq_list row8431 row8431_ordered, row8431_values]
def row8443 : SortedNatTree.Tree := (.node 13033 (.node 11119 (.node 10111 (.node 9349 (.node 9001 .empty .empty) .empty) (.node 10711 .empty .empty)) (.node 11719 (.node 11491 .empty .empty) (.node 13009 .empty .empty))) (.node 15649 (.node 14281 (.node 13633 (.node 13339 .empty .empty) .empty) (.node 14293 .empty .empty)) (.node 15991 (.node 15877 .empty .empty) (.node 16831 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8443_ordered : SortedNatTree.ordered row8443 = true := by decide
set_option maxRecDepth 4096 in
theorem row8443_values : SortedNatTree.values row8443 = Data.row8443 := rfl
theorem row8443_checked : SortedNatTree.contains row8443 = Data.row8443.contains := by
  rw [SortedNatTree.contains_eq_list row8443 row8443_ordered, row8443_values]
def row8447 : SortedNatTree.Tree := (.node 14303 (.node 12437 (.node 9431 (.node 9173 (.node 8699 .empty .empty) .empty) (.node 11717 (.node 10247 .empty .empty) .empty)) (.node 13763 (.node 13691 (.node 12917 .empty .empty) .empty) (.node 14243 .empty .empty))) (.node 15881 (.node 14957 (.node 14831 (.node 14723 .empty .empty) .empty) (.node 15647 (.node 15077 .empty .empty) .empty)) (.node 16673 (.node 16217 (.node 16001 .empty .empty) .empty) (.node 17099 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8447_ordered : SortedNatTree.ordered row8447 = true := by decide
set_option maxRecDepth 4096 in
theorem row8447_values : SortedNatTree.values row8447 = Data.row8447 := rfl
theorem row8447_checked : SortedNatTree.contains row8447 = Data.row8447.contains := by
  rw [SortedNatTree.contains_eq_list row8447 row8447_ordered, row8447_values]
def row8461 : SortedNatTree.Tree := (.node 12301 (.node 9949 (.node 9091 (.node 8893 .empty .empty) (.node 9817 .empty .empty)) (.node 10597 (.node 10477 .empty .empty) (.node 12073 .empty .empty))) (.node 15349 (.node 15139 (.node 13963 .empty .empty) (.node 15289 .empty .empty)) (.node 15913 (.node 15727 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8461_ordered : SortedNatTree.ordered row8461 = true := by decide
set_option maxRecDepth 4096 in
theorem row8461_values : SortedNatTree.values row8461 = Data.row8461 := rfl
theorem row8461_checked : SortedNatTree.contains row8461 = Data.row8461.contains := by
  rw [SortedNatTree.contains_eq_list row8461 row8461_ordered, row8461_values]
def row8467 : SortedNatTree.Tree := (.node 12583 (.node 10069 (.node 9151 (.node 8689 .empty .empty) (.node 9697 .empty .empty)) (.node 10903 (.node 10729 .empty .empty) .empty)) (.node 14983 (.node 13669 (.node 13597 .empty .empty) .empty) (.node 15937 (.node 15607 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8467_ordered : SortedNatTree.ordered row8467 = true := by decide
set_option maxRecDepth 4096 in
theorem row8467_values : SortedNatTree.values row8467 = Data.row8467 := rfl
theorem row8467_checked : SortedNatTree.contains row8467 = Data.row8467.contains := by
  rw [SortedNatTree.contains_eq_list row8467 row8467_ordered, row8467_values]
def row8501 : SortedNatTree.Tree := (.node 15131 (.node 13463 (.node 9719 (.node 8951 .empty .empty) (.node 12401 .empty .empty)) (.node 14081 (.node 13613 .empty .empty) (.node 14831 .empty .empty))) (.node 16097 (.node 15641 (.node 15467 .empty .empty) (.node 16067 .empty .empty)) (.node 16349 (.node 16217 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8501_ordered : SortedNatTree.ordered row8501 = true := by decide
set_option maxRecDepth 4096 in
theorem row8501_values : SortedNatTree.values row8501 = Data.row8501 := rfl
theorem row8501_checked : SortedNatTree.contains row8501 = Data.row8501.contains := by
  rw [SortedNatTree.contains_eq_list row8501 row8501_ordered, row8501_values]
def row8513 : SortedNatTree.Tree := (.node 13163 (.node 10331 (.node 9227 (.node 8831 (.node 8747 .empty .empty) .empty) (.node 10289 (.node 10007 .empty .empty) .empty)) (.node 11831 (.node 11399 (.node 10709 .empty .empty) .empty) (.node 12821 .empty .empty))) (.node 15299 (.node 14699 (.node 13901 (.node 13469 .empty .empty) .empty) (.node 15053 .empty .empty)) (.node 17333 (.node 17189 (.node 15791 .empty .empty) .empty) (.node 17387 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8513_ordered : SortedNatTree.ordered row8513 = true := by decide
set_option maxRecDepth 4096 in
theorem row8513_values : SortedNatTree.values row8513 = Data.row8513 := rfl
theorem row8513_checked : SortedNatTree.contains row8513 = Data.row8513.contains := by
  rw [SortedNatTree.contains_eq_list row8513 row8513_ordered, row8513_values]
def row8521 : SortedNatTree.Tree := (.node 10009 (.node 9133 (.node 8821 (.node 8677 .empty .empty) (.node 8923 .empty .empty)) (.node 9613 (.node 9157 .empty .empty) (.node 9619 .empty .empty))) (.node 13417 (.node 11311 (.node 10837 .empty .empty) (.node 12889 .empty .empty)) (.node 15139 (.node 14503 .empty .empty) (.node 15241 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8521_ordered : SortedNatTree.ordered row8521 = true := by decide
set_option maxRecDepth 4096 in
theorem row8521_values : SortedNatTree.values row8521 = Data.row8521 := rfl
theorem row8521_checked : SortedNatTree.contains row8521 = Data.row8521.contains := by
  rw [SortedNatTree.contains_eq_list row8521 row8521_ordered, row8521_values]
def row8527 : SortedNatTree.Tree := (.node 12253 (.node 9613 (.node 9433 (.node 8929 .empty .empty) .empty) (.node 9907 .empty .empty)) (.node 16741 (.node 14557 (.node 13093 .empty .empty) .empty) (.node 17209 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8527_ordered : SortedNatTree.ordered row8527 = true := by decide
set_option maxRecDepth 4096 in
theorem row8527_values : SortedNatTree.values row8527 = Data.row8527 := rfl
theorem row8527_checked : SortedNatTree.contains row8527 = Data.row8527.contains := by
  rw [SortedNatTree.contains_eq_list row8527 row8527_ordered, row8527_values]
def row8537 : SortedNatTree.Tree := (.node 11867 (.node 9629 (.node 8999 (.node 8963 (.node 8753 .empty .empty) .empty) (.node 9551 (.node 9377 .empty .empty) .empty)) (.node 11279 (.node 11093 (.node 10391 .empty .empty) .empty) (.node 11519 .empty .empty))) (.node 15017 (.node 14639 (.node 14009 (.node 12323 .empty .empty) .empty) (.node 14747 .empty .empty)) (.node 16979 (.node 15881 (.node 15761 .empty .empty) .empty) (.node 17351 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8537_ordered : SortedNatTree.ordered row8537 = true := by decide
set_option maxRecDepth 4096 in
theorem row8537_values : SortedNatTree.values row8537 = Data.row8537 := rfl
theorem row8537_checked : SortedNatTree.contains row8537 = Data.row8537.contains := by
  rw [SortedNatTree.contains_eq_list row8537 row8537_ordered, row8537_values]
def row8539 : SortedNatTree.Tree := (.node 15061 (.node 12853 (.node 9901 (.node 9343 .empty .empty) .empty) (.node 14563 .empty .empty)) (.node 17107 (.node 15607 .empty .empty) (.node 17359 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8539_ordered : SortedNatTree.ordered row8539 = true := by decide
set_option maxRecDepth 4096 in
theorem row8539_values : SortedNatTree.values row8539 = Data.row8539 := rfl
theorem row8539_checked : SortedNatTree.contains row8539 = Data.row8539.contains := by
  rw [SortedNatTree.contains_eq_list row8539 row8539_ordered, row8539_values]
def row8543 : SortedNatTree.Tree := (.node 12119 (.node 10169 (.node 9791 (.node 9281 (.node 8609 .empty .empty) .empty) (.node 9803 .empty .empty)) (.node 11351 (.node 10859 (.node 10463 .empty .empty) .empty) (.node 11519 .empty .empty))) (.node 13907 (.node 13337 (.node 12413 (.node 12323 .empty .empty) .empty) (.node 13751 .empty .empty)) (.node 14969 (.node 14207 .empty .empty) (.node 16937 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8543_ordered : SortedNatTree.ordered row8543 = true := by decide
set_option maxRecDepth 4096 in
theorem row8543_values : SortedNatTree.values row8543 = Data.row8543 := rfl
theorem row8543_checked : SortedNatTree.contains row8543 = Data.row8543.contains := by
  rw [SortedNatTree.contains_eq_list row8543 row8543_ordered, row8543_values]
def row8563 : SortedNatTree.Tree := (.node 14029 (.node 11311 (.node 10141 (.node 9343 .empty .empty) (.node 10243 .empty .empty)) (.node 11887 (.node 11383 .empty .empty) (.node 12157 .empty .empty))) (.node 15583 (.node 14653 (.node 14551 .empty .empty) (.node 15091 .empty .empty)) (.node 16183 (.node 15667 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8563_ordered : SortedNatTree.ordered row8563 = true := by decide
set_option maxRecDepth 4096 in
theorem row8563_values : SortedNatTree.values row8563 = Data.row8563 := rfl
theorem row8563_checked : SortedNatTree.contains row8563 = Data.row8563.contains := by
  rw [SortedNatTree.contains_eq_list row8563 row8563_ordered, row8563_values]
def row8573 : SortedNatTree.Tree := (.node 11549 (.node 9623 (.node 9497 (.node 9161 (.node 8753 .empty .empty) .empty) (.node 9539 .empty .empty)) (.node 10163 (.node 10079 .empty .empty) (.node 11471 .empty .empty))) (.node 14957 (.node 12641 (.node 12251 .empty .empty) (.node 14411 .empty .empty)) (.node 15791 (.node 15749 .empty .empty) (.node 16871 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8573_ordered : SortedNatTree.ordered row8573 = true := by decide
set_option maxRecDepth 4096 in
theorem row8573_values : SortedNatTree.values row8573 = Data.row8573 := rfl
theorem row8573_checked : SortedNatTree.contains row8573 = Data.row8573.contains := by
  rw [SortedNatTree.contains_eq_list row8573 row8573_ordered, row8573_values]
def row8581 : SortedNatTree.Tree := (.node 12919 (.node 11257 (.node 9241 (.node 9157 (.node 9127 .empty .empty) .empty) (.node 10399 (.node 9931 .empty .empty) .empty)) (.node 12097 (.node 11887 (.node 11719 .empty .empty) .empty) (.node 12907 (.node 12163 .empty .empty) .empty))) (.node 14071 (.node 13411 (.node 13339 (.node 13297 .empty .empty) .empty) (.node 14011 (.node 13933 .empty .empty) .empty)) (.node 15889 (.node 15679 (.node 14143 .empty .empty) .empty) (.node 16993 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8581_ordered : SortedNatTree.ordered row8581 = true := by decide
set_option maxRecDepth 4096 in
theorem row8581_values : SortedNatTree.values row8581 = Data.row8581 := rfl
theorem row8581_checked : SortedNatTree.contains row8581 = Data.row8581.contains := by
  rw [SortedNatTree.contains_eq_list row8581 row8581_ordered, row8581_values]
def row8597 : SortedNatTree.Tree := (.node 11399 (.node 9587 (.node 9227 (.node 9209 (.node 9029 .empty .empty) .empty) (.node 9239 .empty .empty)) (.node 10289 (.node 10193 (.node 9767 .empty .empty) .empty) (.node 10343 .empty .empty))) (.node 13457 (.node 13259 (.node 13187 (.node 12671 .empty .empty) .empty) (.node 13451 .empty .empty)) (.node 16619 (.node 16607 (.node 14519 .empty .empty) .empty) (.node 16661 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8597_ordered : SortedNatTree.ordered row8597 = true := by decide
set_option maxRecDepth 4096 in
theorem row8597_values : SortedNatTree.values row8597 = Data.row8597 := rfl
theorem row8597_checked : SortedNatTree.contains row8597 = Data.row8597.contains := by
  rw [SortedNatTree.contains_eq_list row8597 row8597_ordered, row8597_values]
def row8599 : SortedNatTree.Tree := (.node 12577 (.node 10357 (.node 9931 (.node 8623 .empty .empty) .empty) (.node 12457 (.node 12253 .empty .empty) .empty)) (.node 15307 (.node 14431 (.node 12841 .empty .empty) .empty) (.node 17299 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8599_ordered : SortedNatTree.ordered row8599 = true := by decide
set_option maxRecDepth 4096 in
theorem row8599_values : SortedNatTree.values row8599 = Data.row8599 := rfl
theorem row8599_checked : SortedNatTree.contains row8599 = Data.row8599.contains := by
  rw [SortedNatTree.contains_eq_list row8599 row8599_ordered, row8599_values]
def row8609 : SortedNatTree.Tree := (.node 12347 (.node 10781 (.node 9743 (.node 9491 .empty .empty) (.node 10601 .empty .empty)) (.node 11927 (.node 10937 .empty .empty) .empty)) (.node 14543 (.node 12893 (.node 12809 .empty .empty) (.node 14207 .empty .empty)) (.node 16547 (.node 15527 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8609_ordered : SortedNatTree.ordered row8609 = true := by decide
set_option maxRecDepth 4096 in
theorem row8609_values : SortedNatTree.values row8609 = Data.row8609 := rfl
theorem row8609_checked : SortedNatTree.contains row8609 = Data.row8609.contains := by
  rw [SortedNatTree.contains_eq_list row8609 row8609_ordered, row8609_values]
def row8623 : SortedNatTree.Tree := (.node 12421 (.node 10333 (.node 9619 (.node 9091 (.node 8863 .empty .empty) .empty) (.node 10093 .empty .empty)) (.node 10909 (.node 10723 (.node 10597 .empty .empty) .empty) (.node 11119 .empty .empty))) (.node 15217 (.node 13789 (.node 13513 (.node 13147 .empty .empty) .empty) (.node 14437 .empty .empty)) (.node 16033 (.node 15541 .empty .empty) (.node 17041 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8623_ordered : SortedNatTree.ordered row8623 = true := by decide
set_option maxRecDepth 4096 in
theorem row8623_values : SortedNatTree.values row8623 = Data.row8623 := rfl
theorem row8623_checked : SortedNatTree.contains row8623 = Data.row8623.contains := by
  rw [SortedNatTree.contains_eq_list row8623 row8623_ordered, row8623_values]
def row8627 : SortedNatTree.Tree := (.node 14543 (.node 9533 (.node 9173 (.node 8861 .empty .empty) (.node 9257 .empty .empty)) (.node 11939 (.node 10313 .empty .empty) (.node 13121 .empty .empty))) (.node 15263 (.node 15101 (.node 14639 .empty .empty) (.node 15137 .empty .empty)) (.node 16931 (.node 15731 .empty .empty) (.node 17159 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8627_ordered : SortedNatTree.ordered row8627 = true := by decide
set_option maxRecDepth 4096 in
theorem row8627_values : SortedNatTree.values row8627 = Data.row8627 := rfl
theorem row8627_checked : SortedNatTree.contains row8627 = Data.row8627.contains := by
  rw [SortedNatTree.contains_eq_list row8627 row8627_ordered, row8627_values]
def row8629 : SortedNatTree.Tree := (.node 14029 (.node 12577 (.node 11059 (.node 10267 .empty .empty) .empty) (.node 13267 (.node 12703 .empty .empty) .empty)) (.node 15583 (.node 15451 (.node 14869 .empty .empty) .empty) (.node 16981 (.node 16633 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8629_ordered : SortedNatTree.ordered row8629 = true := by decide
set_option maxRecDepth 4096 in
theorem row8629_values : SortedNatTree.values row8629 = Data.row8629 := rfl
theorem row8629_checked : SortedNatTree.contains row8629 = Data.row8629.contains := by
  rw [SortedNatTree.contains_eq_list row8629 row8629_ordered, row8629_values]
def row8641 : SortedNatTree.Tree := (.node 13963 (.node 11491 (.node 10531 (.node 9649 (.node 8719 .empty .empty) .empty) (.node 10729 .empty .empty)) (.node 12511 (.node 12157 .empty .empty) (.node 13267 .empty .empty))) (.node 15787 (.node 15061 (.node 14821 .empty .empty) (.node 15271 .empty .empty)) (.node 16657 (.node 16249 .empty .empty) (.node 17299 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8641_ordered : SortedNatTree.ordered row8641 = true := by decide
set_option maxRecDepth 4096 in
theorem row8641_values : SortedNatTree.values row8641 = Data.row8641 := rfl
theorem row8641_checked : SortedNatTree.contains row8641 = Data.row8641.contains := by
  rw [SortedNatTree.contains_eq_list row8641 row8641_ordered, row8641_values]
def row8647 : SortedNatTree.Tree := (.node 13159 (.node 9091 (.node 8929 (.node 8713 .empty .empty) (.node 9007 .empty .empty)) (.node 11503 (.node 9403 .empty .empty) (.node 11863 .empty .empty))) (.node 16231 (.node 13963 (.node 13417 .empty .empty) (.node 15427 .empty .empty)) (.node 16981 (.node 16273 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8647_ordered : SortedNatTree.ordered row8647 = true := by decide
set_option maxRecDepth 4096 in
theorem row8647_values : SortedNatTree.values row8647 = Data.row8647 := rfl
theorem row8647_checked : SortedNatTree.contains row8647 = Data.row8647.contains := by
  rw [SortedNatTree.contains_eq_list row8647 row8647_ordered, row8647_values]
def row8663 : SortedNatTree.Tree := (.node 12203 (.node 10337 (.node 9629 (.node 9461 .empty .empty) .empty) (.node 11909 (.node 10343 .empty .empty) .empty)) (.node 14717 (.node 13127 (.node 12437 .empty .empty) .empty) (.node 15803 (.node 15101 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8663_ordered : SortedNatTree.ordered row8663 = true := by decide
set_option maxRecDepth 4096 in
theorem row8663_values : SortedNatTree.values row8663 = Data.row8663 := rfl
theorem row8663_checked : SortedNatTree.contains row8663 = Data.row8663.contains := by
  rw [SortedNatTree.contains_eq_list row8663 row8663_ordered, row8663_values]
def row8669 : SortedNatTree.Tree := (.node 12401 (.node 10211 (.node 9551 (.node 9203 (.node 8867 .empty .empty) .empty) (.node 9803 .empty .empty)) (.node 11657 (.node 11489 .empty .empty) (.node 11867 .empty .empty))) (.node 14549 (.node 13127 (.node 12983 .empty .empty) (.node 14051 .empty .empty)) (.node 16007 (.node 14627 .empty .empty) (.node 17099 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8669_ordered : SortedNatTree.ordered row8669 = true := by decide
set_option maxRecDepth 4096 in
theorem row8669_values : SortedNatTree.values row8669 = Data.row8669 := rfl
theorem row8669_checked : SortedNatTree.contains row8669 = Data.row8669.contains := by
  rw [SortedNatTree.contains_eq_list row8669 row8669_ordered, row8669_values]
def row8677 : SortedNatTree.Tree := (.node 12037 (.node 10453 (.node 9661 (.node 9613 (.node 9103 .empty .empty) .empty) (.node 10399 .empty .empty)) (.node 11701 (.node 11443 .empty .empty) (.node 11833 .empty .empty))) (.node 14347 (.node 13009 (.node 12253 .empty .empty) (.node 13879 .empty .empty)) (.node 15859 (.node 15661 .empty .empty) (.node 17029 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8677_ordered : SortedNatTree.ordered row8677 = true := by decide
set_option maxRecDepth 4096 in
theorem row8677_values : SortedNatTree.values row8677 = Data.row8677 := rfl
theorem row8677_checked : SortedNatTree.contains row8677 = Data.row8677.contains := by
  rw [SortedNatTree.contains_eq_list row8677 row8677_ordered, row8677_values]
def row8681 : SortedNatTree.Tree := (.node 14867 (.node 11789 (.node 11243 (.node 11117 (.node 10301 .empty .empty) .empty) (.node 11411 .empty .empty)) (.node 13691 (.node 12911 .empty .empty) (.node 14153 .empty .empty))) (.node 15581 (.node 15359 (.node 15161 .empty .empty) (.node 15383 .empty .empty)) (.node 16187 (.node 16103 .empty .empty) (.node 16703 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8681_ordered : SortedNatTree.ordered row8681 = true := by decide
set_option maxRecDepth 4096 in
theorem row8681_values : SortedNatTree.values row8681 = Data.row8681 := rfl
theorem row8681_checked : SortedNatTree.contains row8681 = Data.row8681.contains := by
  rw [SortedNatTree.contains_eq_list row8681 row8681_ordered, row8681_values]
def row8689 : SortedNatTree.Tree := (.node 13411 (.node 10243 (.node 9403 (.node 8887 .empty .empty) (.node 9781 .empty .empty)) (.node 12301 (.node 11257 .empty .empty) .empty)) (.node 14887 (.node 13879 (.node 13789 .empty .empty) .empty) (.node 16981 (.node 15013 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8689_ordered : SortedNatTree.ordered row8689 = true := by decide
set_option maxRecDepth 4096 in
theorem row8689_values : SortedNatTree.values row8689 = Data.row8689 := rfl
theorem row8689_checked : SortedNatTree.contains row8689 = Data.row8689.contains := by
  rw [SortedNatTree.contains_eq_list row8689 row8689_ordered, row8689_values]
def row8693 : SortedNatTree.Tree := (.node 13619 (.node 9371 (.node 9341 (.node 8783 .empty .empty) .empty) (.node 10067 .empty .empty)) (.node 15581 (.node 14783 .empty .empty) (.node 17291 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8693_ordered : SortedNatTree.ordered row8693 = true := by decide
set_option maxRecDepth 4096 in
theorem row8693_values : SortedNatTree.values row8693 = Data.row8693 := rfl
theorem row8693_checked : SortedNatTree.contains row8693 = Data.row8693.contains := by
  rw [SortedNatTree.contains_eq_list row8693 row8693_ordered, row8693_values]
def row8699 : SortedNatTree.Tree := (.node 11909 (.node 9929 (.node 9323 (.node 9281 (.node 8963 .empty .empty) .empty) (.node 9467 .empty .empty)) (.node 11213 (.node 10667 (.node 10103 .empty .empty) .empty) (.node 11807 .empty .empty))) (.node 13883 (.node 13007 (.node 12983 (.node 12503 .empty .empty) .empty) (.node 13313 .empty .empty)) (.node 16829 (.node 15461 (.node 14699 .empty .empty) .empty) (.node 17117 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8699_ordered : SortedNatTree.ordered row8699 = true := by decide
set_option maxRecDepth 4096 in
theorem row8699_values : SortedNatTree.values row8699 = Data.row8699 := rfl
theorem row8699_checked : SortedNatTree.contains row8699 = Data.row8699.contains := by
  rw [SortedNatTree.contains_eq_list row8699 row8699_ordered, row8699_values]
def row8707 : SortedNatTree.Tree := (.node 12421 (.node 10837 (.node 10177 (.node 9397 (.node 8839 .empty .empty) .empty) (.node 10723 .empty .empty)) (.node 11971 (.node 11239 (.node 11131 .empty .empty) .empty) (.node 12253 .empty .empty))) (.node 13873 (.node 13297 (.node 13033 (.node 12967 .empty .empty) .empty) (.node 13537 .empty .empty)) (.node 16633 (.node 15667 .empty .empty) (.node 16981 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8707_ordered : SortedNatTree.ordered row8707 = true := by decide
set_option maxRecDepth 4096 in
theorem row8707_values : SortedNatTree.values row8707 = Data.row8707 := rfl
theorem row8707_checked : SortedNatTree.contains row8707 = Data.row8707.contains := by
  rw [SortedNatTree.contains_eq_list row8707 row8707_ordered, row8707_values]
def row8713 : SortedNatTree.Tree := (.node 13879 (.node 11719 (.node 9859 (.node 9439 (.node 9319 .empty .empty) .empty) (.node 10753 .empty .empty)) (.node 12601 (.node 11971 .empty .empty) (.node 13873 .empty .empty))) (.node 15349 (.node 14713 (.node 14449 (.node 14071 .empty .empty) .empty) (.node 15241 .empty .empty)) (.node 16921 (.node 16603 .empty .empty) (.node 17011 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8713_ordered : SortedNatTree.ordered row8713 = true := by decide
set_option maxRecDepth 4096 in
theorem row8713_values : SortedNatTree.values row8713 = Data.row8713 := rfl
theorem row8713_checked : SortedNatTree.contains row8713 = Data.row8713.contains := by
  rw [SortedNatTree.contains_eq_list row8713 row8713_ordered, row8713_values]
def row8719 : SortedNatTree.Tree := (.node 14737 (.node 11923 (.node 11119 (.node 9901 (.node 9649 .empty .empty) (.node 10987 .empty .empty)) (.node 11839 (.node 11701 .empty .empty) .empty)) (.node 12889 (.node 12703 (.node 12409 .empty .empty) .empty) (.node 14341 (.node 13723 .empty .empty) .empty))) (.node 15817 (.node 15391 (.node 15271 (.node 14779 .empty .empty) (.node 15313 .empty .empty)) (.node 15739 (.node 15559 .empty .empty) .empty)) (.node 16231 (.node 16189 (.node 15991 .empty .empty) .empty) (.node 16993 (.node 16699 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row8719_ordered : SortedNatTree.ordered row8719 = true := by decide
set_option maxRecDepth 4096 in
theorem row8719_values : SortedNatTree.values row8719 = Data.row8719 := rfl
theorem row8719_checked : SortedNatTree.contains row8719 = Data.row8719.contains := by
  rw [SortedNatTree.contains_eq_list row8719 row8719_ordered, row8719_values]
def row8731 : SortedNatTree.Tree := (.node 10369 (.node 9949 (.node 9391 (.node 9151 (.node 9007 .empty .empty) .empty) (.node 9433 .empty .empty)) (.node 10177 (.node 10099 (.node 10093 .empty .empty) .empty) (.node 10333 .empty .empty))) (.node 13711 (.node 12157 (.node 11299 (.node 10723 .empty .empty) .empty) (.node 12973 .empty .empty)) (.node 14929 (.node 14887 (.node 14503 .empty .empty) .empty) (.node 15013 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8731_ordered : SortedNatTree.ordered row8731 = true := by decide
set_option maxRecDepth 4096 in
theorem row8731_values : SortedNatTree.values row8731 = Data.row8731 := rfl
theorem row8731_checked : SortedNatTree.contains row8731 = Data.row8731.contains := by
  rw [SortedNatTree.contains_eq_list row8731 row8731_ordered, row8731_values]
def row8737 : SortedNatTree.Tree := (.node 14683 (.node 11299 (.node 9883 (.node 9739 (.node 9151 .empty .empty) .empty) (.node 10069 .empty .empty)) (.node 12517 (.node 11953 (.node 11491 .empty .empty) .empty) (.node 14533 .empty .empty))) (.node 15607 (.node 15091 (.node 15061 (.node 14767 .empty .empty) .empty) (.node 15121 .empty .empty)) (.node 16087 (.node 15649 .empty .empty) (.node 17293 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8737_ordered : SortedNatTree.ordered row8737 = true := by decide
set_option maxRecDepth 4096 in
theorem row8737_values : SortedNatTree.values row8737 = Data.row8737 := rfl
theorem row8737_checked : SortedNatTree.contains row8737 = Data.row8737.contains := by
  rw [SortedNatTree.contains_eq_list row8737 row8737_ordered, row8737_values]
def row8741 : SortedNatTree.Tree := (.node 13313 (.node 11471 (.node 9677 (.node 9239 .empty .empty) (.node 10151 .empty .empty)) (.node 11789 (.node 11633 .empty .empty) (.node 13109 .empty .empty))) (.node 15461 (.node 14783 (.node 14771 .empty .empty) (.node 15233 .empty .empty)) (.node 16301 (.node 15731 .empty .empty) (.node 17117 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8741_ordered : SortedNatTree.ordered row8741 = true := by decide
set_option maxRecDepth 4096 in
theorem row8741_values : SortedNatTree.values row8741 = Data.row8741 := rfl
theorem row8741_checked : SortedNatTree.contains row8741 = Data.row8741.contains := by
  rw [SortedNatTree.contains_eq_list row8741 row8741_ordered, row8741_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
