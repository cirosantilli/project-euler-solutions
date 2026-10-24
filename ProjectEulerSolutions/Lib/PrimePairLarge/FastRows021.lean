import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row11117 : SortedNatTree.Tree := (.node 13523 (.node 13121 (.node 12347 .empty .empty) .empty) (.node 14369 (.node 13709 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11117_ordered : SortedNatTree.ordered row11117 = true := by decide
set_option maxRecDepth 4096 in
theorem row11117_values : SortedNatTree.values row11117 = Data.row11117 := rfl
theorem row11117_checked : SortedNatTree.contains row11117 = Data.row11117.contains := by
  rw [SortedNatTree.contains_eq_list row11117 row11117_ordered, row11117_values]
def row11119 : SortedNatTree.Tree := (.node 12757 (.node 12589 (.node 11467 .empty .empty) .empty) (.node 13477 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11119_ordered : SortedNatTree.ordered row11119 = true := by decide
set_option maxRecDepth 4096 in
theorem row11119_values : SortedNatTree.values row11119 = Data.row11119 := rfl
theorem row11119_checked : SortedNatTree.contains row11119 = Data.row11119.contains := by
  rw [SortedNatTree.contains_eq_list row11119 row11119_ordered, row11119_values]
def row11131 : SortedNatTree.Tree := (.node 13999 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11131_ordered : SortedNatTree.ordered row11131 = true := by decide
set_option maxRecDepth 4096 in
theorem row11131_values : SortedNatTree.values row11131 = Data.row11131 := rfl
theorem row11131_checked : SortedNatTree.contains row11131 = Data.row11131.contains := by
  rw [SortedNatTree.contains_eq_list row11131 row11131_ordered, row11131_values]
def row11149 : SortedNatTree.Tree := (.node 12619 (.node 12583 (.node 11383 .empty .empty) .empty) (.node 14767 (.node 12979 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11149_ordered : SortedNatTree.ordered row11149 = true := by decide
set_option maxRecDepth 4096 in
theorem row11149_values : SortedNatTree.values row11149 = Data.row11149 := rfl
theorem row11149_checked : SortedNatTree.contains row11149 = Data.row11149.contains := by
  rw [SortedNatTree.contains_eq_list row11149 row11149_ordered, row11149_values]
def row11159 : SortedNatTree.Tree := (.node 12923 (.node 12437 (.node 12227 (.node 11633 .empty .empty) .empty) (.node 12659 .empty .empty)) (.node 14447 (.node 13859 .empty .empty) (.node 14717 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11159_ordered : SortedNatTree.ordered row11159 = true := by decide
set_option maxRecDepth 4096 in
theorem row11159_values : SortedNatTree.values row11159 = Data.row11159 := rfl
theorem row11159_checked : SortedNatTree.contains row11159 = Data.row11159.contains := by
  rw [SortedNatTree.contains_eq_list row11159 row11159_ordered, row11159_values]
def row11161 : SortedNatTree.Tree := (.node 13693 (.node 13309 (.node 12799 .empty .empty) .empty) (.node 14437 (.node 14071 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11161_ordered : SortedNatTree.ordered row11161 = true := by decide
set_option maxRecDepth 4096 in
theorem row11161_values : SortedNatTree.values row11161 = Data.row11161 := rfl
theorem row11161_checked : SortedNatTree.contains row11161 = Data.row11161.contains := by
  rw [SortedNatTree.contains_eq_list row11161 row11161_ordered, row11161_values]
def row11171 : SortedNatTree.Tree := (.node 12413 (.node 11321 (.node 11279 .empty .empty) .empty) (.node 14051 (.node 13967 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11171_ordered : SortedNatTree.ordered row11171 = true := by decide
set_option maxRecDepth 4096 in
theorem row11171_values : SortedNatTree.values row11171 = Data.row11171 := rfl
theorem row11171_checked : SortedNatTree.contains row11171 = Data.row11171.contains := by
  rw [SortedNatTree.contains_eq_list row11171 row11171_ordered, row11171_values]
def row11173 : SortedNatTree.Tree := (.node 13063 (.node 12739 (.node 11719 .empty .empty) (.node 12967 .empty .empty)) (.node 14797 (.node 13789 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11173_ordered : SortedNatTree.ordered row11173 = true := by decide
set_option maxRecDepth 4096 in
theorem row11173_values : SortedNatTree.values row11173 = Data.row11173 := rfl
theorem row11173_checked : SortedNatTree.contains row11173 = Data.row11173.contains := by
  rw [SortedNatTree.contains_eq_list row11173 row11173_ordered, row11173_values]
def row11177 : SortedNatTree.Tree := (.node 11807 (.node 11369 (.node 11351 .empty .empty) .empty) (.node 14519 (.node 13757 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11177_ordered : SortedNatTree.ordered row11177 = true := by decide
set_option maxRecDepth 4096 in
theorem row11177_values : SortedNatTree.values row11177 = Data.row11177 := rfl
theorem row11177_checked : SortedNatTree.contains row11177 = Data.row11177.contains := by
  rw [SortedNatTree.contains_eq_list row11177 row11177_ordered, row11177_values]
def row11197 : SortedNatTree.Tree := (.node 12457 (.node 12163 (.node 12043 (.node 11941 .empty .empty) .empty) (.node 12409 .empty .empty)) (.node 13711 (.node 13297 (.node 13093 .empty .empty) .empty) (.node 13879 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11197_ordered : SortedNatTree.ordered row11197 = true := by decide
set_option maxRecDepth 4096 in
theorem row11197_values : SortedNatTree.values row11197 = Data.row11197 := rfl
theorem row11197_checked : SortedNatTree.contains row11197 = Data.row11197.contains := by
  rw [SortedNatTree.contains_eq_list row11197 row11197_ordered, row11197_values]
def row11213 : SortedNatTree.Tree := (.node 13001 (.node 12497 (.node 11813 .empty .empty) .empty) (.node 14771 (.node 13763 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11213_ordered : SortedNatTree.ordered row11213 = true := by decide
set_option maxRecDepth 4096 in
theorem row11213_values : SortedNatTree.values row11213 = Data.row11213 := rfl
theorem row11213_checked : SortedNatTree.contains row11213 = Data.row11213.contains := by
  rw [SortedNatTree.contains_eq_list row11213 row11213_ordered, row11213_values]
def row11239 : SortedNatTree.Tree := (.node 14437 (.node 12157 (.node 11779 .empty .empty) .empty) (.node 14629 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11239_ordered : SortedNatTree.ordered row11239 = true := by decide
set_option maxRecDepth 4096 in
theorem row11239_values : SortedNatTree.values row11239 = Data.row11239 := rfl
theorem row11239_checked : SortedNatTree.contains row11239 = Data.row11239.contains := by
  rw [SortedNatTree.contains_eq_list row11239 row11239_ordered, row11239_values]
def row11243 : SortedNatTree.Tree := (.node 12713 (.node 11789 (.node 11621 .empty .empty) (.node 12659 .empty .empty)) (.node 14783 (.node 13763 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11243_ordered : SortedNatTree.ordered row11243 = true := by decide
set_option maxRecDepth 4096 in
theorem row11243_values : SortedNatTree.values row11243 = Data.row11243 := rfl
theorem row11243_checked : SortedNatTree.contains row11243 = Data.row11243.contains := by
  rw [SortedNatTree.contains_eq_list row11243 row11243_ordered, row11243_values]
def row11251 : SortedNatTree.Tree := (.node 13009 (.node 12619 (.node 11941 .empty .empty) (.node 12907 .empty .empty)) (.node 14503 (.node 14293 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11251_ordered : SortedNatTree.ordered row11251 = true := by decide
set_option maxRecDepth 4096 in
theorem row11251_values : SortedNatTree.values row11251 = Data.row11251 := rfl
theorem row11251_checked : SortedNatTree.contains row11251 = Data.row11251.contains := by
  rw [SortedNatTree.contains_eq_list row11251 row11251_ordered, row11251_values]
def row11257 : SortedNatTree.Tree := (.node 14173 (.node 11863 (.node 11497 .empty .empty) (.node 13177 .empty .empty)) (.node 14593 (.node 14281 .empty .empty) (.node 14731 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11257_ordered : SortedNatTree.ordered row11257 = true := by decide
set_option maxRecDepth 4096 in
theorem row11257_values : SortedNatTree.values row11257 = Data.row11257 := rfl
theorem row11257_checked : SortedNatTree.contains row11257 = Data.row11257.contains := by
  rw [SortedNatTree.contains_eq_list row11257 row11257_ordered, row11257_values]
def row11261 : SortedNatTree.Tree := (.node 12503 (.node 12401 (.node 12161 .empty .empty) .empty) (.node 14243 (.node 13697 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11261_ordered : SortedNatTree.ordered row11261 = true := by decide
set_option maxRecDepth 4096 in
theorem row11261_values : SortedNatTree.values row11261 = Data.row11261 := rfl
theorem row11261_checked : SortedNatTree.contains row11261 = Data.row11261.contains := by
  rw [SortedNatTree.contains_eq_list row11261 row11261_ordered, row11261_values]
def row11273 : SortedNatTree.Tree := (.node 13463 (.node 12113 (.node 11777 (.node 11579 .empty .empty) .empty) (.node 13337 .empty .empty)) (.node 14633 (.node 13829 .empty .empty) (.node 14759 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11273_ordered : SortedNatTree.ordered row11273 = true := by decide
set_option maxRecDepth 4096 in
theorem row11273_values : SortedNatTree.values row11273 = Data.row11273 := rfl
theorem row11273_checked : SortedNatTree.contains row11273 = Data.row11273.contains := by
  rw [SortedNatTree.contains_eq_list row11273 row11273_ordered, row11273_values]
def row11279 : SortedNatTree.Tree := (.node 13691 (.node 13613 (.node 11549 .empty .empty) .empty) (.node 14747 (.node 13781 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11279_ordered : SortedNatTree.ordered row11279 = true := by decide
set_option maxRecDepth 4096 in
theorem row11279_values : SortedNatTree.values row11279 = Data.row11279 := rfl
theorem row11279_checked : SortedNatTree.contains row11279 = Data.row11279.contains := by
  rw [SortedNatTree.contains_eq_list row11279 row11279_ordered, row11279_values]
def row11287 : SortedNatTree.Tree := (.node 12919 (.node 12457 .empty .empty) (.node 14281 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11287_ordered : SortedNatTree.ordered row11287 = true := by decide
set_option maxRecDepth 4096 in
theorem row11287_values : SortedNatTree.values row11287 = Data.row11287 := rfl
theorem row11287_checked : SortedNatTree.contains row11287 = Data.row11287.contains := by
  rw [SortedNatTree.contains_eq_list row11287 row11287_ordered, row11287_values]
def row11299 : SortedNatTree.Tree := (.node 13009 (.node 12799 (.node 12757 .empty .empty) .empty) (.node 14551 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11299_ordered : SortedNatTree.ordered row11299 = true := by decide
set_option maxRecDepth 4096 in
theorem row11299_values : SortedNatTree.values row11299 = Data.row11299 := rfl
theorem row11299_checked : SortedNatTree.contains row11299 = Data.row11299.contains := by
  rw [SortedNatTree.contains_eq_list row11299 row11299_ordered, row11299_values]
def row11311 : SortedNatTree.Tree := (.node 13099 (.node 12487 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11311_ordered : SortedNatTree.ordered row11311 = true := by decide
set_option maxRecDepth 4096 in
theorem row11311_values : SortedNatTree.values row11311 = Data.row11311 := rfl
theorem row11311_checked : SortedNatTree.contains row11311 = Data.row11311.contains := by
  rw [SortedNatTree.contains_eq_list row11311 row11311_ordered, row11311_values]
def row11317 : SortedNatTree.Tree := (.node 12721 (.node 12157 (.node 11551 .empty .empty) .empty) (.node 14593 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11317_ordered : SortedNatTree.ordered row11317 = true := by decide
set_option maxRecDepth 4096 in
theorem row11317_values : SortedNatTree.values row11317 = Data.row11317 := rfl
theorem row11317_checked : SortedNatTree.contains row11317 = Data.row11317.contains := by
  rw [SortedNatTree.contains_eq_list row11317 row11317_ordered, row11317_values]
def row11321 : SortedNatTree.Tree := (.node 13883 (.node 12497 (.node 11987 .empty .empty) .empty) (.node 13907 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11321_ordered : SortedNatTree.ordered row11321 = true := by decide
set_option maxRecDepth 4096 in
theorem row11321_values : SortedNatTree.values row11321 = Data.row11321 := rfl
theorem row11321_checked : SortedNatTree.contains row11321 = Data.row11321.contains := by
  rw [SortedNatTree.contains_eq_list row11321 row11321_ordered, row11321_values]
def row11329 : SortedNatTree.Tree := (.node 13147 (.node 12421 (.node 12097 .empty .empty) (.node 13003 .empty .empty)) (.node 13633 (.node 13381 .empty .empty) (.node 14389 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11329_ordered : SortedNatTree.ordered row11329 = true := by decide
set_option maxRecDepth 4096 in
theorem row11329_values : SortedNatTree.values row11329 = Data.row11329 := rfl
theorem row11329_checked : SortedNatTree.contains row11329 = Data.row11329.contains := by
  rw [SortedNatTree.contains_eq_list row11329 row11329_ordered, row11329_values]
def row11351 : SortedNatTree.Tree := (.node 13451 (.node 12659 (.node 11597 .empty .empty) (.node 12821 .empty .empty)) (.node 14411 (.node 14087 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11351_ordered : SortedNatTree.ordered row11351 = true := by decide
set_option maxRecDepth 4096 in
theorem row11351_values : SortedNatTree.values row11351 = Data.row11351 := rfl
theorem row11351_checked : SortedNatTree.contains row11351 = Data.row11351.contains := by
  rw [SortedNatTree.contains_eq_list row11351 row11351_ordered, row11351_values]
def row11353 : SortedNatTree.Tree := (.node 12433 (.node 11941 (.node 11719 .empty .empty) (.node 12289 .empty .empty)) (.node 13687 (.node 12601 .empty .empty) (.node 13789 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11353_ordered : SortedNatTree.ordered row11353 = true := by decide
set_option maxRecDepth 4096 in
theorem row11353_values : SortedNatTree.values row11353 = Data.row11353 := rfl
theorem row11353_checked : SortedNatTree.contains row11353 = Data.row11353.contains := by
  rw [SortedNatTree.contains_eq_list row11353 row11353_ordered, row11353_values]
def row11369 : SortedNatTree.Tree := (.node 13103 (.node 13007 (.node 12377 (.node 11681 .empty .empty) .empty) (.node 13049 .empty .empty)) (.node 14591 (.node 14447 .empty .empty) (.node 14657 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11369_ordered : SortedNatTree.ordered row11369 = true := by decide
set_option maxRecDepth 4096 in
theorem row11369_values : SortedNatTree.values row11369 = Data.row11369 := rfl
theorem row11369_checked : SortedNatTree.contains row11369 = Data.row11369.contains := by
  rw [SortedNatTree.contains_eq_list row11369 row11369_ordered, row11369_values]
def row11383 : SortedNatTree.Tree := (.node 14071 (.node 12739 (.node 11467 .empty .empty) .empty) (.node 14461 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11383_ordered : SortedNatTree.ordered row11383 = true := by decide
set_option maxRecDepth 4096 in
theorem row11383_values : SortedNatTree.values row11383 = Data.row11383 := rfl
theorem row11383_checked : SortedNatTree.contains row11383 = Data.row11383.contains := by
  rw [SortedNatTree.contains_eq_list row11383 row11383_ordered, row11383_values]
def row11393 : SortedNatTree.Tree := (.node 13697 (.node 11633 (.node 11471 .empty .empty) (.node 12893 .empty .empty)) (.node 14633 (.node 13709 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11393_ordered : SortedNatTree.ordered row11393 = true := by decide
set_option maxRecDepth 4096 in
theorem row11393_values : SortedNatTree.values row11393 = Data.row11393 := rfl
theorem row11393_checked : SortedNatTree.contains row11393 = Data.row11393.contains := by
  rw [SortedNatTree.contains_eq_list row11393 row11393_ordered, row11393_values]
def row11399 : SortedNatTree.Tree := (.node 12911 (.node 12161 .empty .empty) (.node 14249 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11399_ordered : SortedNatTree.ordered row11399 = true := by decide
set_option maxRecDepth 4096 in
theorem row11399_values : SortedNatTree.values row11399 = Data.row11399 := rfl
theorem row11399_checked : SortedNatTree.contains row11399 = Data.row11399.contains := by
  rw [SortedNatTree.contains_eq_list row11399 row11399_ordered, row11399_values]
def row11411 : SortedNatTree.Tree := (.node 12941 (.node 12281 .empty .empty) (.node 13907 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11411_ordered : SortedNatTree.ordered row11411 = true := by decide
set_option maxRecDepth 4096 in
theorem row11411_values : SortedNatTree.values row11411 = Data.row11411 := rfl
theorem row11411_checked : SortedNatTree.contains row11411 = Data.row11411.contains := by
  rw [SortedNatTree.contains_eq_list row11411 row11411_ordered, row11411_values]
def row11423 : SortedNatTree.Tree := (.node 12959 (.node 12473 (.node 12401 (.node 11579 .empty .empty) .empty) (.node 12653 .empty .empty)) (.node 13577 (.node 13553 .empty .empty) (.node 14057 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11423_ordered : SortedNatTree.ordered row11423 = true := by decide
set_option maxRecDepth 4096 in
theorem row11423_values : SortedNatTree.values row11423 = Data.row11423 := rfl
theorem row11423_checked : SortedNatTree.contains row11423 = Data.row11423.contains := by
  rw [SortedNatTree.contains_eq_list row11423 row11423_ordered, row11423_values]
def row11437 : SortedNatTree.Tree := (.node 13789 (.node 12301 (.node 11971 .empty .empty) (.node 13729 .empty .empty)) (.node 14323 (.node 14251 .empty .empty) (.node 14431 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11437_ordered : SortedNatTree.ordered row11437 = true := by decide
set_option maxRecDepth 4096 in
theorem row11437_values : SortedNatTree.values row11437 = Data.row11437 := rfl
theorem row11437_checked : SortedNatTree.contains row11437 = Data.row11437.contains := by
  rw [SortedNatTree.contains_eq_list row11437 row11437_ordered, row11437_values]
def row11443 : SortedNatTree.Tree := (.node 14341 (.node 14011 (.node 12553 .empty .empty) .empty) (.node 14449 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11443_ordered : SortedNatTree.ordered row11443 = true := by decide
set_option maxRecDepth 4096 in
theorem row11443_values : SortedNatTree.values row11443 = Data.row11443 := rfl
theorem row11443_checked : SortedNatTree.contains row11443 = Data.row11443.contains := by
  rw [SortedNatTree.contains_eq_list row11443 row11443_ordered, row11443_values]
def row11447 : SortedNatTree.Tree := (.node 12071 (.node 11927 (.node 11783 (.node 11483 .empty .empty) .empty) (.node 12011 .empty .empty)) (.node 13859 (.node 13709 (.node 12203 .empty .empty) .empty) (.node 14369 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11447_ordered : SortedNatTree.ordered row11447 = true := by decide
set_option maxRecDepth 4096 in
theorem row11447_values : SortedNatTree.values row11447 = Data.row11447 := rfl
theorem row11447_checked : SortedNatTree.contains row11447 = Data.row11447.contains := by
  rw [SortedNatTree.contains_eq_list row11447 row11447_ordered, row11447_values]
def row11467 : SortedNatTree.Tree := (.node 12487 (.node 12421 (.node 11887 .empty .empty) .empty) (.node 12979 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11467_ordered : SortedNatTree.ordered row11467 = true := by decide
set_option maxRecDepth 4096 in
theorem row11467_values : SortedNatTree.values row11467 = Data.row11467 := rfl
theorem row11467_checked : SortedNatTree.contains row11467 = Data.row11467.contains := by
  rw [SortedNatTree.contains_eq_list row11467 row11467_ordered, row11467_values]
def row11471 : SortedNatTree.Tree := (.node 14057 (.node 12647 (.node 12569 .empty .empty) (.node 13679 .empty .empty)) (.node 14369 (.node 14249 .empty .empty) (.node 14537 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11471_ordered : SortedNatTree.ordered row11471 = true := by decide
set_option maxRecDepth 4096 in
theorem row11471_values : SortedNatTree.values row11471 = Data.row11471 := rfl
theorem row11471_checked : SortedNatTree.contains row11471 = Data.row11471.contains := by
  rw [SortedNatTree.contains_eq_list row11471 row11471_ordered, row11471_values]
def row11483 : SortedNatTree.Tree := (.node 13421 (.node 12611 (.node 12041 .empty .empty) .empty) (.node 13619 (.node 13499 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11483_ordered : SortedNatTree.ordered row11483 = true := by decide
set_option maxRecDepth 4096 in
theorem row11483_values : SortedNatTree.values row11483 = Data.row11483 := rfl
theorem row11483_checked : SortedNatTree.contains row11483 = Data.row11483.contains := by
  rw [SortedNatTree.contains_eq_list row11483 row11483_ordered, row11483_values]
def row11489 : SortedNatTree.Tree := (.node 13127 (.node 11783 (.node 11597 .empty .empty) (.node 12899 .empty .empty)) (.node 13709 (.node 13229 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11489_ordered : SortedNatTree.ordered row11489 = true := by decide
set_option maxRecDepth 4096 in
theorem row11489_values : SortedNatTree.values row11489 = Data.row11489 := rfl
theorem row11489_checked : SortedNatTree.contains row11489 = Data.row11489.contains := by
  rw [SortedNatTree.contains_eq_list row11489 row11489_ordered, row11489_values]
def row11491 : SortedNatTree.Tree := (.node 13003 (.node 12409 .empty .empty) (.node 13291 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11491_ordered : SortedNatTree.ordered row11491 = true := by decide
set_option maxRecDepth 4096 in
theorem row11491_values : SortedNatTree.values row11491 = Data.row11491 := rfl
theorem row11491_checked : SortedNatTree.contains row11491 = Data.row11491.contains := by
  rw [SortedNatTree.contains_eq_list row11491 row11491_ordered, row11491_values]
def row11497 : SortedNatTree.Tree := (.node 12541 (.node 11677 .empty .empty) (.node 12583 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11497_ordered : SortedNatTree.ordered row11497 = true := by decide
set_option maxRecDepth 4096 in
theorem row11497_values : SortedNatTree.values row11497 = Data.row11497 := rfl
theorem row11497_checked : SortedNatTree.contains row11497 = Data.row11497.contains := by
  rw [SortedNatTree.contains_eq_list row11497 row11497_ordered, row11497_values]
def row11503 : SortedNatTree.Tree := (.node 12697 (.node 12613 (.node 12577 .empty .empty) .empty) (.node 13921 (.node 13627 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11503_ordered : SortedNatTree.ordered row11503 = true := by decide
set_option maxRecDepth 4096 in
theorem row11503_values : SortedNatTree.values row11503 = Data.row11503 := rfl
theorem row11503_checked : SortedNatTree.contains row11503 = Data.row11503.contains := by
  rw [SortedNatTree.contains_eq_list row11503 row11503_ordered, row11503_values]
def row11519 : SortedNatTree.Tree := (.node 12239 (.node 11927 (.node 11681 (.node 11633 .empty .empty) .empty) (.node 12149 .empty .empty)) (.node 13331 (.node 12689 (.node 12569 .empty .empty) .empty) (.node 14447 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11519_ordered : SortedNatTree.ordered row11519 = true := by decide
set_option maxRecDepth 4096 in
theorem row11519_values : SortedNatTree.values row11519 = Data.row11519 := rfl
theorem row11519_checked : SortedNatTree.contains row11519 = Data.row11519.contains := by
  rw [SortedNatTree.contains_eq_list row11519 row11519_ordered, row11519_values]
def row11527 : SortedNatTree.Tree := (.node 13567 (.node 12703 (.node 12073 .empty .empty) .empty) (.node 14449 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11527_ordered : SortedNatTree.ordered row11527 = true := by decide
set_option maxRecDepth 4096 in
theorem row11527_values : SortedNatTree.values row11527 = Data.row11527 := rfl
theorem row11527_checked : SortedNatTree.contains row11527 = Data.row11527.contains := by
  rw [SortedNatTree.contains_eq_list row11527 row11527_ordered, row11527_values]
def row11549 : SortedNatTree.Tree := (.node 12917 (.node 12107 (.node 11783 .empty .empty) .empty) (.node 13577 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11549_ordered : SortedNatTree.ordered row11549 = true := by decide
set_option maxRecDepth 4096 in
theorem row11549_values : SortedNatTree.values row11549 = Data.row11549 := rfl
theorem row11549_checked : SortedNatTree.contains row11549 = Data.row11549.contains := by
  rw [SortedNatTree.contains_eq_list row11549 row11549_ordered, row11549_values]
def row11551 : SortedNatTree.Tree := (.node 13147 (.node 11887 (.node 11833 .empty .empty) .empty) (.node 14251 (.node 14173 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11551_ordered : SortedNatTree.ordered row11551 = true := by decide
set_option maxRecDepth 4096 in
theorem row11551_values : SortedNatTree.values row11551 = Data.row11551 := rfl
theorem row11551_checked : SortedNatTree.contains row11551 = Data.row11551.contains := by
  rw [SortedNatTree.contains_eq_list row11551 row11551_ordered, row11551_values]
def row11579 : SortedNatTree.Tree := (.node 12263 (.node 12011 (.node 11789 .empty .empty) .empty) (.node 12893 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11579_ordered : SortedNatTree.ordered row11579 = true := by decide
set_option maxRecDepth 4096 in
theorem row11579_values : SortedNatTree.values row11579 = Data.row11579 := rfl
theorem row11579_checked : SortedNatTree.contains row11579 = Data.row11579.contains := by
  rw [SortedNatTree.contains_eq_list row11579 row11579_ordered, row11579_values]
def row11587 : SortedNatTree.Tree := (.node 12589 (.node 12007 .empty .empty) (.node 12889 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11587_ordered : SortedNatTree.ordered row11587 = true := by decide
set_option maxRecDepth 4096 in
theorem row11587_values : SortedNatTree.values row11587 = Data.row11587 := rfl
theorem row11587_checked : SortedNatTree.contains row11587 = Data.row11587.contains := by
  rw [SortedNatTree.contains_eq_list row11587 row11587_ordered, row11587_values]
def row11593 : SortedNatTree.Tree := (.node 13567 (.node 11833 .empty .empty) (.node 14323 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11593_ordered : SortedNatTree.ordered row11593 = true := by decide
set_option maxRecDepth 4096 in
theorem row11593_values : SortedNatTree.values row11593 = Data.row11593 := rfl
theorem row11593_checked : SortedNatTree.contains row11593 = Data.row11593.contains := by
  rw [SortedNatTree.contains_eq_list row11593 row11593_ordered, row11593_values]
def row11597 : SortedNatTree.Tree := (.node 12713 (.node 12269 (.node 11897 .empty .empty) (.node 12413 .empty .empty)) (.node 14009 (.node 13619 .empty .empty) (.node 14327 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11597_ordered : SortedNatTree.ordered row11597 = true := by decide
set_option maxRecDepth 4096 in
theorem row11597_values : SortedNatTree.values row11597 = Data.row11597 := rfl
theorem row11597_checked : SortedNatTree.contains row11597 = Data.row11597.contains := by
  rw [SortedNatTree.contains_eq_list row11597 row11597_ordered, row11597_values]
def row11617 : SortedNatTree.Tree := (.node 13669 (.node 12157 (.node 12043 .empty .empty) .empty) (.node 14293 (.node 13681 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11617_ordered : SortedNatTree.ordered row11617 = true := by decide
set_option maxRecDepth 4096 in
theorem row11617_values : SortedNatTree.values row11617 = Data.row11617 := rfl
theorem row11617_checked : SortedNatTree.contains row11617 = Data.row11617.contains := by
  rw [SortedNatTree.contains_eq_list row11617 row11617_ordered, row11617_values]
def row11621 : SortedNatTree.Tree := (.node 14321 (.node 14159 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11621_ordered : SortedNatTree.ordered row11621 = true := by decide
set_option maxRecDepth 4096 in
theorem row11621_values : SortedNatTree.values row11621 = Data.row11621 := rfl
theorem row11621_checked : SortedNatTree.contains row11621 = Data.row11621.contains := by
  rw [SortedNatTree.contains_eq_list row11621 row11621_ordered, row11621_values]
def row11633 : SortedNatTree.Tree := (.node 12251 (.node 11927 .empty .empty) (.node 12713 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11633_ordered : SortedNatTree.ordered row11633 = true := by decide
set_option maxRecDepth 4096 in
theorem row11633_values : SortedNatTree.values row11633 = Data.row11633 := rfl
theorem row11633_checked : SortedNatTree.contains row11633 = Data.row11633.contains := by
  rw [SortedNatTree.contains_eq_list row11633 row11633_ordered, row11633_values]
def row11657 : SortedNatTree.Tree := (.node 11699 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11657_ordered : SortedNatTree.ordered row11657 = true := by decide
set_option maxRecDepth 4096 in
theorem row11657_values : SortedNatTree.values row11657 = Data.row11657 := rfl
theorem row11657_checked : SortedNatTree.contains row11657 = Data.row11657.contains := by
  rw [SortedNatTree.contains_eq_list row11657 row11657_ordered, row11657_values]
def row11677 : SortedNatTree.Tree := (.node 13567 (.node 12757 (.node 12379 (.node 12097 .empty .empty) .empty) (.node 13441 .empty .empty)) (.node 13999 (.node 13669 .empty .empty) (.node 14143 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11677_ordered : SortedNatTree.ordered row11677 = true := by decide
set_option maxRecDepth 4096 in
theorem row11677_values : SortedNatTree.values row11677 = Data.row11677 := rfl
theorem row11677_checked : SortedNatTree.contains row11677 = Data.row11677.contains := by
  rw [SortedNatTree.contains_eq_list row11677 row11677_ordered, row11677_values]
def row11681 : SortedNatTree.Tree := (.node 13553 (.node 12899 (.node 11897 .empty .empty) (.node 13241 .empty .empty)) (.node 14159 (.node 13613 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11681_ordered : SortedNatTree.ordered row11681 = true := by decide
set_option maxRecDepth 4096 in
theorem row11681_values : SortedNatTree.values row11681 = Data.row11681 := rfl
theorem row11681_checked : SortedNatTree.contains row11681 = Data.row11681.contains := by
  rw [SortedNatTree.contains_eq_list row11681 row11681_ordered, row11681_values]
def row11689 : SortedNatTree.Tree := (.node 12763 (.node 12517 (.node 11701 .empty .empty) .empty) (.node 13033 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11689_ordered : SortedNatTree.ordered row11689 = true := by decide
set_option maxRecDepth 4096 in
theorem row11689_values : SortedNatTree.values row11689 = Data.row11689 := rfl
theorem row11689_checked : SortedNatTree.contains row11689 = Data.row11689.contains := by
  rw [SortedNatTree.contains_eq_list row11689 row11689_ordered, row11689_values]
def row11699 : SortedNatTree.Tree := (.node 12959 .empty .empty)
set_option maxRecDepth 4096 in
theorem row11699_ordered : SortedNatTree.ordered row11699 = true := by decide
set_option maxRecDepth 4096 in
theorem row11699_values : SortedNatTree.values row11699 = Data.row11699 := rfl
theorem row11699_checked : SortedNatTree.contains row11699 = Data.row11699.contains := by
  rw [SortedNatTree.contains_eq_list row11699 row11699_ordered, row11699_values]
def row11701 : SortedNatTree.Tree := (.node 13597 (.node 12781 (.node 11779 .empty .empty) .empty) (.node 13759 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11701_ordered : SortedNatTree.ordered row11701 = true := by decide
set_option maxRecDepth 4096 in
theorem row11701_values : SortedNatTree.values row11701 = Data.row11701 := rfl
theorem row11701_checked : SortedNatTree.contains row11701 = Data.row11701.contains := by
  rw [SortedNatTree.contains_eq_list row11701 row11701_ordered, row11701_values]
def row11717 : SortedNatTree.Tree := (.node 12821 (.node 12251 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row11717_ordered : SortedNatTree.ordered row11717 = true := by decide
set_option maxRecDepth 4096 in
theorem row11717_values : SortedNatTree.values row11717 = Data.row11717 := rfl
theorem row11717_checked : SortedNatTree.contains row11717 = Data.row11717.contains := by
  rw [SortedNatTree.contains_eq_list row11717 row11717_ordered, row11717_values]
def row11719 : SortedNatTree.Tree := (.node 11971 (.node 11821 .empty .empty) (.node 12619 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11719_ordered : SortedNatTree.ordered row11719 = true := by decide
set_option maxRecDepth 4096 in
theorem row11719_values : SortedNatTree.values row11719 = Data.row11719 := rfl
theorem row11719_checked : SortedNatTree.contains row11719 = Data.row11719.contains := by
  rw [SortedNatTree.contains_eq_list row11719 row11719_ordered, row11719_values]
def row11731 : SortedNatTree.Tree := (.node 13147 (.node 12613 (.node 11743 .empty .empty) (.node 12637 .empty .empty)) (.node 13999 (.node 13729 .empty .empty) (.node 14083 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11731_ordered : SortedNatTree.ordered row11731 = true := by decide
set_option maxRecDepth 4096 in
theorem row11731_values : SortedNatTree.values row11731 = Data.row11731 := rfl
theorem row11731_checked : SortedNatTree.contains row11731 = Data.row11731.contains := by
  rw [SortedNatTree.contains_eq_list row11731 row11731_ordered, row11731_values]
def row11743 : SortedNatTree.Tree := (.node 13009 (.node 12487 (.node 11953 .empty .empty) .empty) (.node 13669 (.node 13177 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11743_ordered : SortedNatTree.ordered row11743 = true := by decide
set_option maxRecDepth 4096 in
theorem row11743_values : SortedNatTree.values row11743 = Data.row11743 := rfl
theorem row11743_checked : SortedNatTree.contains row11743 = Data.row11743.contains := by
  rw [SortedNatTree.contains_eq_list row11743 row11743_ordered, row11743_values]
def row11777 : SortedNatTree.Tree := (.node 12917 (.node 12119 (.node 11867 .empty .empty) (.node 12203 .empty .empty)) (.node 13679 (.node 13331 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11777_ordered : SortedNatTree.ordered row11777 = true := by decide
set_option maxRecDepth 4096 in
theorem row11777_values : SortedNatTree.values row11777 = Data.row11777 := rfl
theorem row11777_checked : SortedNatTree.contains row11777 = Data.row11777.contains := by
  rw [SortedNatTree.contains_eq_list row11777 row11777_ordered, row11777_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
