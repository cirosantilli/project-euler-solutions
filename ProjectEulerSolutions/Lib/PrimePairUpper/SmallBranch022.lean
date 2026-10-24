import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates871 : List Nat := [6899, 7841]
set_option maxRecDepth 4096 in
theorem rootCandidates871_checked : candidates edge 5 0 6791 tail872 best.bound = rootCandidates871 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates870 : List Nat := [6823, 6841, 7069, 7237, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates870_checked : candidates edge 5 0 6781 tail871 best.bound = rootCandidates870 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext q; rfl
  rw [row]
  decide

def innerTail870_5 : List Nat := []
theorem innerScan870_5 : scan edge (search edge 3) 4 [6781] 6781 innerTail870_5 best = best := rfl

def innerTail870_4 : List Nat := 8161 :: innerTail870_5
def innerCandidates870_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates870_4_checked : candidates edge 4 6781 8161 innerTail870_5 best.bound = innerCandidates870_4 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan870_4 : scan edge (search edge 3) 4 [6781] 6781 innerTail870_4 best = best := by
  rw [innerTail870_4, scan, if_neg (by decide), innerCandidates870_4_checked]
  change scan edge (search edge 3) 4 [6781] 6781 innerTail870_5
    (if 3 ≤ innerCandidates870_4.length then search edge 3 [6781, 8161] innerCandidates870_4 14942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan870_5

def innerTail870_3 : List Nat := 7237 :: innerTail870_4
def innerCandidates870_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates870_3_checked : candidates edge 4 6781 7237 innerTail870_4 best.bound = innerCandidates870_3 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan870_3 : scan edge (search edge 3) 4 [6781] 6781 innerTail870_3 best = best := by
  rw [innerTail870_3, scan, if_neg (by decide), innerCandidates870_3_checked]
  change scan edge (search edge 3) 4 [6781] 6781 innerTail870_4
    (if 3 ≤ innerCandidates870_3.length then search edge 3 [6781, 7237] innerCandidates870_3 14018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan870_4

def innerTail870_2 : List Nat := 7069 :: innerTail870_3
def innerCandidates870_2 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates870_2_checked : candidates edge 4 6781 7069 innerTail870_3 best.bound = innerCandidates870_2 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan870_2 : scan edge (search edge 3) 4 [6781] 6781 innerTail870_2 best = best := by
  rw [innerTail870_2, scan, if_neg (by decide), innerCandidates870_2_checked]
  change scan edge (search edge 3) 4 [6781] 6781 innerTail870_3
    (if 3 ≤ innerCandidates870_2.length then search edge 3 [6781, 7069] innerCandidates870_2 13850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan870_3

def innerTail870_1 : List Nat := 6841 :: innerTail870_2
def innerCandidates870_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates870_1_checked : candidates edge 4 6781 6841 innerTail870_2 best.bound = innerCandidates870_1 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan870_1 : scan edge (search edge 3) 4 [6781] 6781 innerTail870_1 best = best := by
  rw [innerTail870_1, scan, if_neg (by decide), innerCandidates870_1_checked]
  change scan edge (search edge 3) 4 [6781] 6781 innerTail870_2
    (if 3 ≤ innerCandidates870_1.length then search edge 3 [6781, 6841] innerCandidates870_1 13622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan870_2

def innerTail870_0 : List Nat := 6823 :: innerTail870_1
def innerCandidates870_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates870_0_checked : candidates edge 4 6781 6823 innerTail870_1 best.bound = innerCandidates870_0 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan870_0 : scan edge (search edge 3) 4 [6781] 6781 innerTail870_0 best = best := by
  rw [innerTail870_0, scan, if_neg (by decide), innerCandidates870_0_checked]
  change scan edge (search edge 3) 4 [6781] 6781 innerTail870_1
    (if 3 ≤ innerCandidates870_0.length then search edge 3 [6781, 6823] innerCandidates870_0 13604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan870_1

theorem rootBranch870 : search edge 4 [6781] rootCandidates870 6781 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6781] 6781 innerTail870_0 best = best
  exact innerScan870_0

def rootCandidates869 : List Nat := [7487, 7517, 7823, 8231, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates869_checked : candidates edge 5 0 6779 tail870 best.bound = rootCandidates869 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext q; rfl
  rw [row]
  decide

def innerTail869_5 : List Nat := []
theorem innerScan869_5 : scan edge (search edge 3) 4 [6779] 6779 innerTail869_5 best = best := rfl

def innerTail869_4 : List Nat := 8369 :: innerTail869_5
def innerCandidates869_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates869_4_checked : candidates edge 4 6779 8369 innerTail869_5 best.bound = innerCandidates869_4 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan869_4 : scan edge (search edge 3) 4 [6779] 6779 innerTail869_4 best = best := by
  rw [innerTail869_4, scan, if_neg (by decide), innerCandidates869_4_checked]
  change scan edge (search edge 3) 4 [6779] 6779 innerTail869_5
    (if 3 ≤ innerCandidates869_4.length then search edge 3 [6779, 8369] innerCandidates869_4 15148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan869_5

def innerTail869_3 : List Nat := 8231 :: innerTail869_4
def innerCandidates869_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates869_3_checked : candidates edge 4 6779 8231 innerTail869_4 best.bound = innerCandidates869_3 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan869_3 : scan edge (search edge 3) 4 [6779] 6779 innerTail869_3 best = best := by
  rw [innerTail869_3, scan, if_neg (by decide), innerCandidates869_3_checked]
  change scan edge (search edge 3) 4 [6779] 6779 innerTail869_4
    (if 3 ≤ innerCandidates869_3.length then search edge 3 [6779, 8231] innerCandidates869_3 15010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan869_4

def innerTail869_2 : List Nat := 7823 :: innerTail869_3
def innerCandidates869_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates869_2_checked : candidates edge 4 6779 7823 innerTail869_3 best.bound = innerCandidates869_2 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan869_2 : scan edge (search edge 3) 4 [6779] 6779 innerTail869_2 best = best := by
  rw [innerTail869_2, scan, if_neg (by decide), innerCandidates869_2_checked]
  change scan edge (search edge 3) 4 [6779] 6779 innerTail869_3
    (if 3 ≤ innerCandidates869_2.length then search edge 3 [6779, 7823] innerCandidates869_2 14602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan869_3

def innerTail869_1 : List Nat := 7517 :: innerTail869_2
def innerCandidates869_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates869_1_checked : candidates edge 4 6779 7517 innerTail869_2 best.bound = innerCandidates869_1 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan869_1 : scan edge (search edge 3) 4 [6779] 6779 innerTail869_1 best = best := by
  rw [innerTail869_1, scan, if_neg (by decide), innerCandidates869_1_checked]
  change scan edge (search edge 3) 4 [6779] 6779 innerTail869_2
    (if 3 ≤ innerCandidates869_1.length then search edge 3 [6779, 7517] innerCandidates869_1 14296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan869_2

def innerTail869_0 : List Nat := 7487 :: innerTail869_1
def innerCandidates869_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates869_0_checked : candidates edge 4 6779 7487 innerTail869_1 best.bound = innerCandidates869_0 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan869_0 : scan edge (search edge 3) 4 [6779] 6779 innerTail869_0 best = best := by
  rw [innerTail869_0, scan, if_neg (by decide), innerCandidates869_0_checked]
  change scan edge (search edge 3) 4 [6779] 6779 innerTail869_1
    (if 3 ≤ innerCandidates869_0.length then search edge 3 [6779, 7487] innerCandidates869_0 14266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan869_1

theorem rootBranch869 : search edge 4 [6779] rootCandidates869 6779 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6779] 6779 innerTail869_0 best = best
  exact innerScan869_0

def rootCandidates868 : List Nat := [7333, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates868_checked : candidates edge 5 0 6763 tail869 best.bound = rootCandidates868 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates867 : List Nat := [7229, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates867_checked : candidates edge 5 0 6761 tail868 best.bound = rootCandidates867 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates866 : List Nat := [6779, 7451]
set_option maxRecDepth 4096 in
theorem rootCandidates866_checked : candidates edge 5 0 6737 tail867 best.bound = rootCandidates866 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates865 : List Nat := [7321, 7789, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates865_checked : candidates edge 5 0 6733 tail866 best.bound = rootCandidates865 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates864 : List Nat := [7349, 7523, 7589, 7607, 7829, 8147, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates864_checked : candidates edge 5 0 6719 tail865 best.bound = rootCandidates864 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext q; rfl
  rw [row]
  decide

def innerTail864_7 : List Nat := []
theorem innerScan864_7 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_7 best = best := rfl

def innerTail864_6 : List Nat := 8237 :: innerTail864_7
def innerCandidates864_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates864_6_checked : candidates edge 4 6719 8237 innerTail864_7 best.bound = innerCandidates864_6 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_6 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_6 best = best := by
  rw [innerTail864_6, scan, if_neg (by decide), innerCandidates864_6_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_7
    (if 3 ≤ innerCandidates864_6.length then search edge 3 [6719, 8237] innerCandidates864_6 14956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_7

def innerTail864_5 : List Nat := 8147 :: innerTail864_6
def innerCandidates864_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates864_5_checked : candidates edge 4 6719 8147 innerTail864_6 best.bound = innerCandidates864_5 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_5 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_5 best = best := by
  rw [innerTail864_5, scan, if_neg (by decide), innerCandidates864_5_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_6
    (if 3 ≤ innerCandidates864_5.length then search edge 3 [6719, 8147] innerCandidates864_5 14866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_6

def innerTail864_4 : List Nat := 7829 :: innerTail864_5
def innerCandidates864_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates864_4_checked : candidates edge 4 6719 7829 innerTail864_5 best.bound = innerCandidates864_4 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_4 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_4 best = best := by
  rw [innerTail864_4, scan, if_neg (by decide), innerCandidates864_4_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_5
    (if 3 ≤ innerCandidates864_4.length then search edge 3 [6719, 7829] innerCandidates864_4 14548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_5

def innerTail864_3 : List Nat := 7607 :: innerTail864_4
def innerCandidates864_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates864_3_checked : candidates edge 4 6719 7607 innerTail864_4 best.bound = innerCandidates864_3 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_3 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_3 best = best := by
  rw [innerTail864_3, scan, if_neg (by decide), innerCandidates864_3_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_4
    (if 3 ≤ innerCandidates864_3.length then search edge 3 [6719, 7607] innerCandidates864_3 14326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_4

def innerTail864_2 : List Nat := 7589 :: innerTail864_3
def innerCandidates864_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates864_2_checked : candidates edge 4 6719 7589 innerTail864_3 best.bound = innerCandidates864_2 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_2 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_2 best = best := by
  rw [innerTail864_2, scan, if_neg (by decide), innerCandidates864_2_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_3
    (if 3 ≤ innerCandidates864_2.length then search edge 3 [6719, 7589] innerCandidates864_2 14308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_3

def innerTail864_1 : List Nat := 7523 :: innerTail864_2
def innerCandidates864_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates864_1_checked : candidates edge 4 6719 7523 innerTail864_2 best.bound = innerCandidates864_1 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_1 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_1 best = best := by
  rw [innerTail864_1, scan, if_neg (by decide), innerCandidates864_1_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_2
    (if 3 ≤ innerCandidates864_1.length then search edge 3 [6719, 7523] innerCandidates864_1 14242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_2

def innerTail864_0 : List Nat := 7349 :: innerTail864_1
def innerCandidates864_0 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates864_0_checked : candidates edge 4 6719 7349 innerTail864_1 best.bound = innerCandidates864_0 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan864_0 : scan edge (search edge 3) 4 [6719] 6719 innerTail864_0 best = best := by
  rw [innerTail864_0, scan, if_neg (by decide), innerCandidates864_0_checked]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_1
    (if 3 ≤ innerCandidates864_0.length then search edge 3 [6719, 7349] innerCandidates864_0 14068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan864_1

theorem rootBranch864 : search edge 4 [6719] rootCandidates864 6719 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6719] 6719 innerTail864_0 best = best
  exact innerScan864_0

end ProjectEulerSolutions.PrimePairUpper.Search
