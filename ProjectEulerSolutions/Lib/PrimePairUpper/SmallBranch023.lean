import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates863 : List Nat := [7369]
set_option maxRecDepth 4096 in
theorem rootCandidates863_checked : candidates edge 5 0 6709 tail864 best.bound = rootCandidates863 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates862 : List Nat := [7213, 7297, 7459, 7603, 7723, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates862_checked : candidates edge 5 0 6703 tail863 best.bound = rootCandidates862 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext q; rfl
  rw [row]
  decide

def innerTail862_6 : List Nat := []
theorem innerScan862_6 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_6 best = best := rfl

def innerTail862_5 : List Nat := 8221 :: innerTail862_6
def innerCandidates862_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates862_5_checked : candidates edge 4 6703 8221 innerTail862_6 best.bound = innerCandidates862_5 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan862_5 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_5 best = best := by
  rw [innerTail862_5, scan, if_neg (by decide), innerCandidates862_5_checked]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_6
    (if 3 ≤ innerCandidates862_5.length then search edge 3 [6703, 8221] innerCandidates862_5 14924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan862_6

def innerTail862_4 : List Nat := 7723 :: innerTail862_5
def innerCandidates862_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates862_4_checked : candidates edge 4 6703 7723 innerTail862_5 best.bound = innerCandidates862_4 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan862_4 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_4 best = best := by
  rw [innerTail862_4, scan, if_neg (by decide), innerCandidates862_4_checked]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_5
    (if 3 ≤ innerCandidates862_4.length then search edge 3 [6703, 7723] innerCandidates862_4 14426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan862_5

def innerTail862_3 : List Nat := 7603 :: innerTail862_4
def innerCandidates862_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates862_3_checked : candidates edge 4 6703 7603 innerTail862_4 best.bound = innerCandidates862_3 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan862_3 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_3 best = best := by
  rw [innerTail862_3, scan, if_neg (by decide), innerCandidates862_3_checked]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_4
    (if 3 ≤ innerCandidates862_3.length then search edge 3 [6703, 7603] innerCandidates862_3 14306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan862_4

def innerTail862_2 : List Nat := 7459 :: innerTail862_3
def innerCandidates862_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates862_2_checked : candidates edge 4 6703 7459 innerTail862_3 best.bound = innerCandidates862_2 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan862_2 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_2 best = best := by
  rw [innerTail862_2, scan, if_neg (by decide), innerCandidates862_2_checked]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_3
    (if 3 ≤ innerCandidates862_2.length then search edge 3 [6703, 7459] innerCandidates862_2 14162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan862_3

def innerTail862_1 : List Nat := 7297 :: innerTail862_2
def innerCandidates862_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates862_1_checked : candidates edge 4 6703 7297 innerTail862_2 best.bound = innerCandidates862_1 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan862_1 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_1 best = best := by
  rw [innerTail862_1, scan, if_neg (by decide), innerCandidates862_1_checked]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_2
    (if 3 ≤ innerCandidates862_1.length then search edge 3 [6703, 7297] innerCandidates862_1 14000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan862_2

def innerTail862_0 : List Nat := 7213 :: innerTail862_1
def innerCandidates862_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates862_0_checked : candidates edge 4 6703 7213 innerTail862_1 best.bound = innerCandidates862_0 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan862_0 : scan edge (search edge 3) 4 [6703] 6703 innerTail862_0 best = best := by
  rw [innerTail862_0, scan, if_neg (by decide), innerCandidates862_0_checked]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_1
    (if 3 ≤ innerCandidates862_0.length then search edge 3 [6703, 7213] innerCandidates862_0 13916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan862_1

theorem rootBranch862 : search edge 4 [6703] rootCandidates862 6703 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6703] 6703 innerTail862_0 best = best
  exact innerScan862_0

def rootCandidates861 : List Nat := [6899, 7019, 8069, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates861_checked : candidates edge 5 0 6701 tail862 best.bound = rootCandidates861 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext q; rfl
  rw [row]
  decide

def innerTail861_4 : List Nat := []
theorem innerScan861_4 : scan edge (search edge 3) 4 [6701] 6701 innerTail861_4 best = best := rfl

def innerTail861_3 : List Nat := 8117 :: innerTail861_4
def innerCandidates861_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates861_3_checked : candidates edge 4 6701 8117 innerTail861_4 best.bound = innerCandidates861_3 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan861_3 : scan edge (search edge 3) 4 [6701] 6701 innerTail861_3 best = best := by
  rw [innerTail861_3, scan, if_neg (by decide), innerCandidates861_3_checked]
  change scan edge (search edge 3) 4 [6701] 6701 innerTail861_4
    (if 3 ≤ innerCandidates861_3.length then search edge 3 [6701, 8117] innerCandidates861_3 14818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan861_4

def innerTail861_2 : List Nat := 8069 :: innerTail861_3
def innerCandidates861_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates861_2_checked : candidates edge 4 6701 8069 innerTail861_3 best.bound = innerCandidates861_2 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan861_2 : scan edge (search edge 3) 4 [6701] 6701 innerTail861_2 best = best := by
  rw [innerTail861_2, scan, if_neg (by decide), innerCandidates861_2_checked]
  change scan edge (search edge 3) 4 [6701] 6701 innerTail861_3
    (if 3 ≤ innerCandidates861_2.length then search edge 3 [6701, 8069] innerCandidates861_2 14770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan861_3

def innerTail861_1 : List Nat := 7019 :: innerTail861_2
def innerCandidates861_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates861_1_checked : candidates edge 4 6701 7019 innerTail861_2 best.bound = innerCandidates861_1 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan861_1 : scan edge (search edge 3) 4 [6701] 6701 innerTail861_1 best = best := by
  rw [innerTail861_1, scan, if_neg (by decide), innerCandidates861_1_checked]
  change scan edge (search edge 3) 4 [6701] 6701 innerTail861_2
    (if 3 ≤ innerCandidates861_1.length then search edge 3 [6701, 7019] innerCandidates861_1 13720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan861_2

def innerTail861_0 : List Nat := 6899 :: innerTail861_1
def innerCandidates861_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates861_0_checked : candidates edge 4 6701 6899 innerTail861_1 best.bound = innerCandidates861_0 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan861_0 : scan edge (search edge 3) 4 [6701] 6701 innerTail861_0 best = best := by
  rw [innerTail861_0, scan, if_neg (by decide), innerCandidates861_0_checked]
  change scan edge (search edge 3) 4 [6701] 6701 innerTail861_1
    (if 3 ≤ innerCandidates861_0.length then search edge 3 [6701, 6899] innerCandidates861_0 13600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan861_1

theorem rootBranch861 : search edge 4 [6701] rootCandidates861 6701 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6701] 6701 innerTail861_0 best = best
  exact innerScan861_0

def rootCandidates860 : List Nat := [6781, 7243, 7507, 7687, 8329, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates860_checked : candidates edge 5 0 6691 tail861 best.bound = rootCandidates860 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext q; rfl
  rw [row]
  decide

def innerTail860_6 : List Nat := []
theorem innerScan860_6 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_6 best = best := rfl

def innerTail860_5 : List Nat := 8377 :: innerTail860_6
def innerCandidates860_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates860_5_checked : candidates edge 4 6691 8377 innerTail860_6 best.bound = innerCandidates860_5 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan860_5 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_5 best = best := by
  rw [innerTail860_5, scan, if_neg (by decide), innerCandidates860_5_checked]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_6
    (if 3 ≤ innerCandidates860_5.length then search edge 3 [6691, 8377] innerCandidates860_5 15068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan860_6

def innerTail860_4 : List Nat := 8329 :: innerTail860_5
def innerCandidates860_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates860_4_checked : candidates edge 4 6691 8329 innerTail860_5 best.bound = innerCandidates860_4 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan860_4 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_4 best = best := by
  rw [innerTail860_4, scan, if_neg (by decide), innerCandidates860_4_checked]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_5
    (if 3 ≤ innerCandidates860_4.length then search edge 3 [6691, 8329] innerCandidates860_4 15020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan860_5

def innerTail860_3 : List Nat := 7687 :: innerTail860_4
def innerCandidates860_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates860_3_checked : candidates edge 4 6691 7687 innerTail860_4 best.bound = innerCandidates860_3 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan860_3 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_3 best = best := by
  rw [innerTail860_3, scan, if_neg (by decide), innerCandidates860_3_checked]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_4
    (if 3 ≤ innerCandidates860_3.length then search edge 3 [6691, 7687] innerCandidates860_3 14378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan860_4

def innerTail860_2 : List Nat := 7507 :: innerTail860_3
def innerCandidates860_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates860_2_checked : candidates edge 4 6691 7507 innerTail860_3 best.bound = innerCandidates860_2 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan860_2 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_2 best = best := by
  rw [innerTail860_2, scan, if_neg (by decide), innerCandidates860_2_checked]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_3
    (if 3 ≤ innerCandidates860_2.length then search edge 3 [6691, 7507] innerCandidates860_2 14198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan860_3

def innerTail860_1 : List Nat := 7243 :: innerTail860_2
def innerCandidates860_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates860_1_checked : candidates edge 4 6691 7243 innerTail860_2 best.bound = innerCandidates860_1 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan860_1 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_1 best = best := by
  rw [innerTail860_1, scan, if_neg (by decide), innerCandidates860_1_checked]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_2
    (if 3 ≤ innerCandidates860_1.length then search edge 3 [6691, 7243] innerCandidates860_1 13934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan860_2

def innerTail860_0 : List Nat := 6781 :: innerTail860_1
def innerCandidates860_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates860_0_checked : candidates edge 4 6691 6781 innerTail860_1 best.bound = innerCandidates860_0 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan860_0 : scan edge (search edge 3) 4 [6691] 6691 innerTail860_0 best = best := by
  rw [innerTail860_0, scan, if_neg (by decide), innerCandidates860_0_checked]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_1
    (if 3 ≤ innerCandidates860_0.length then search edge 3 [6691, 6781] innerCandidates860_0 13472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan860_1

theorem rootBranch860 : search edge 4 [6691] rootCandidates860 6691 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6691] 6691 innerTail860_0 best = best
  exact innerScan860_0

def rootCandidates859 : List Nat := [6779, 7151, 7331, 7487, 7583, 7673, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates859_checked : candidates edge 5 0 6689 tail860 best.bound = rootCandidates859 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext q; rfl
  rw [row]
  decide

def innerTail859_7 : List Nat := []
theorem innerScan859_7 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_7 best = best := rfl

def innerTail859_6 : List Nat := 7949 :: innerTail859_7
def innerCandidates859_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates859_6_checked : candidates edge 4 6689 7949 innerTail859_7 best.bound = innerCandidates859_6 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_6 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_6 best = best := by
  rw [innerTail859_6, scan, if_neg (by decide), innerCandidates859_6_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_7
    (if 3 ≤ innerCandidates859_6.length then search edge 3 [6689, 7949] innerCandidates859_6 14638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_7

def innerTail859_5 : List Nat := 7673 :: innerTail859_6
def innerCandidates859_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates859_5_checked : candidates edge 4 6689 7673 innerTail859_6 best.bound = innerCandidates859_5 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_5 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_5 best = best := by
  rw [innerTail859_5, scan, if_neg (by decide), innerCandidates859_5_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_6
    (if 3 ≤ innerCandidates859_5.length then search edge 3 [6689, 7673] innerCandidates859_5 14362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_6

def innerTail859_4 : List Nat := 7583 :: innerTail859_5
def innerCandidates859_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates859_4_checked : candidates edge 4 6689 7583 innerTail859_5 best.bound = innerCandidates859_4 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_4 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_4 best = best := by
  rw [innerTail859_4, scan, if_neg (by decide), innerCandidates859_4_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_5
    (if 3 ≤ innerCandidates859_4.length then search edge 3 [6689, 7583] innerCandidates859_4 14272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_5

def innerTail859_3 : List Nat := 7487 :: innerTail859_4
def innerCandidates859_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates859_3_checked : candidates edge 4 6689 7487 innerTail859_4 best.bound = innerCandidates859_3 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_3 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_3 best = best := by
  rw [innerTail859_3, scan, if_neg (by decide), innerCandidates859_3_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_4
    (if 3 ≤ innerCandidates859_3.length then search edge 3 [6689, 7487] innerCandidates859_3 14176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_4

def innerTail859_2 : List Nat := 7331 :: innerTail859_3
def innerCandidates859_2 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates859_2_checked : candidates edge 4 6689 7331 innerTail859_3 best.bound = innerCandidates859_2 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_2 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_2 best = best := by
  rw [innerTail859_2, scan, if_neg (by decide), innerCandidates859_2_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_3
    (if 3 ≤ innerCandidates859_2.length then search edge 3 [6689, 7331] innerCandidates859_2 14020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_3

def innerTail859_1 : List Nat := 7151 :: innerTail859_2
def innerCandidates859_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates859_1_checked : candidates edge 4 6689 7151 innerTail859_2 best.bound = innerCandidates859_1 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_1 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_1 best = best := by
  rw [innerTail859_1, scan, if_neg (by decide), innerCandidates859_1_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_2
    (if 3 ≤ innerCandidates859_1.length then search edge 3 [6689, 7151] innerCandidates859_1 13840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_2

def innerTail859_0 : List Nat := 6779 :: innerTail859_1
def innerCandidates859_0 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates859_0_checked : candidates edge 4 6689 6779 innerTail859_1 best.bound = innerCandidates859_0 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan859_0 : scan edge (search edge 3) 4 [6689] 6689 innerTail859_0 best = best := by
  rw [innerTail859_0, scan, if_neg (by decide), innerCandidates859_0_checked]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_1
    (if 3 ≤ innerCandidates859_0.length then search edge 3 [6689, 6779] innerCandidates859_0 13468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan859_1

theorem rootBranch859 : search edge 4 [6689] rootCandidates859 6689 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6689] 6689 innerTail859_0 best = best
  exact innerScan859_0

def rootCandidates858 : List Nat := [6967, 8059, 8161, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates858_checked : candidates edge 5 0 6679 tail859 best.bound = rootCandidates858 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext q; rfl
  rw [row]
  decide

def innerTail858_4 : List Nat := []
theorem innerScan858_4 : scan edge (search edge 3) 4 [6679] 6679 innerTail858_4 best = best := rfl

def innerTail858_3 : List Nat := 8353 :: innerTail858_4
def innerCandidates858_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates858_3_checked : candidates edge 4 6679 8353 innerTail858_4 best.bound = innerCandidates858_3 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan858_3 : scan edge (search edge 3) 4 [6679] 6679 innerTail858_3 best = best := by
  rw [innerTail858_3, scan, if_neg (by decide), innerCandidates858_3_checked]
  change scan edge (search edge 3) 4 [6679] 6679 innerTail858_4
    (if 3 ≤ innerCandidates858_3.length then search edge 3 [6679, 8353] innerCandidates858_3 15032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan858_4

def innerTail858_2 : List Nat := 8161 :: innerTail858_3
def innerCandidates858_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates858_2_checked : candidates edge 4 6679 8161 innerTail858_3 best.bound = innerCandidates858_2 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan858_2 : scan edge (search edge 3) 4 [6679] 6679 innerTail858_2 best = best := by
  rw [innerTail858_2, scan, if_neg (by decide), innerCandidates858_2_checked]
  change scan edge (search edge 3) 4 [6679] 6679 innerTail858_3
    (if 3 ≤ innerCandidates858_2.length then search edge 3 [6679, 8161] innerCandidates858_2 14840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan858_3

def innerTail858_1 : List Nat := 8059 :: innerTail858_2
def innerCandidates858_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates858_1_checked : candidates edge 4 6679 8059 innerTail858_2 best.bound = innerCandidates858_1 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan858_1 : scan edge (search edge 3) 4 [6679] 6679 innerTail858_1 best = best := by
  rw [innerTail858_1, scan, if_neg (by decide), innerCandidates858_1_checked]
  change scan edge (search edge 3) 4 [6679] 6679 innerTail858_2
    (if 3 ≤ innerCandidates858_1.length then search edge 3 [6679, 8059] innerCandidates858_1 14738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan858_2

def innerTail858_0 : List Nat := 6967 :: innerTail858_1
def innerCandidates858_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates858_0_checked : candidates edge 4 6679 6967 innerTail858_1 best.bound = innerCandidates858_0 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan858_0 : scan edge (search edge 3) 4 [6679] 6679 innerTail858_0 best = best := by
  rw [innerTail858_0, scan, if_neg (by decide), innerCandidates858_0_checked]
  change scan edge (search edge 3) 4 [6679] 6679 innerTail858_1
    (if 3 ≤ innerCandidates858_0.length then search edge 3 [6679, 6967] innerCandidates858_0 13646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan858_1

theorem rootBranch858 : search edge 4 [6679] rootCandidates858 6679 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6679] 6679 innerTail858_0 best = best
  exact innerScan858_0

def rootCandidates857 : List Nat := [7351, 7459, 7639, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates857_checked : candidates edge 5 0 6673 tail858 best.bound = rootCandidates857 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext q; rfl
  rw [row]
  decide

def innerTail857_4 : List Nat := []
theorem innerScan857_4 : scan edge (search edge 3) 4 [6673] 6673 innerTail857_4 best = best := rfl

def innerTail857_3 : List Nat := 8293 :: innerTail857_4
def innerCandidates857_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates857_3_checked : candidates edge 4 6673 8293 innerTail857_4 best.bound = innerCandidates857_3 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan857_3 : scan edge (search edge 3) 4 [6673] 6673 innerTail857_3 best = best := by
  rw [innerTail857_3, scan, if_neg (by decide), innerCandidates857_3_checked]
  change scan edge (search edge 3) 4 [6673] 6673 innerTail857_4
    (if 3 ≤ innerCandidates857_3.length then search edge 3 [6673, 8293] innerCandidates857_3 14966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan857_4

def innerTail857_2 : List Nat := 7639 :: innerTail857_3
def innerCandidates857_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates857_2_checked : candidates edge 4 6673 7639 innerTail857_3 best.bound = innerCandidates857_2 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan857_2 : scan edge (search edge 3) 4 [6673] 6673 innerTail857_2 best = best := by
  rw [innerTail857_2, scan, if_neg (by decide), innerCandidates857_2_checked]
  change scan edge (search edge 3) 4 [6673] 6673 innerTail857_3
    (if 3 ≤ innerCandidates857_2.length then search edge 3 [6673, 7639] innerCandidates857_2 14312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan857_3

def innerTail857_1 : List Nat := 7459 :: innerTail857_2
def innerCandidates857_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates857_1_checked : candidates edge 4 6673 7459 innerTail857_2 best.bound = innerCandidates857_1 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan857_1 : scan edge (search edge 3) 4 [6673] 6673 innerTail857_1 best = best := by
  rw [innerTail857_1, scan, if_neg (by decide), innerCandidates857_1_checked]
  change scan edge (search edge 3) 4 [6673] 6673 innerTail857_2
    (if 3 ≤ innerCandidates857_1.length then search edge 3 [6673, 7459] innerCandidates857_1 14132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan857_2

def innerTail857_0 : List Nat := 7351 :: innerTail857_1
def innerCandidates857_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates857_0_checked : candidates edge 4 6673 7351 innerTail857_1 best.bound = innerCandidates857_0 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan857_0 : scan edge (search edge 3) 4 [6673] 6673 innerTail857_0 best = best := by
  rw [innerTail857_0, scan, if_neg (by decide), innerCandidates857_0_checked]
  change scan edge (search edge 3) 4 [6673] 6673 innerTail857_1
    (if 3 ≤ innerCandidates857_0.length then search edge 3 [6673, 7351] innerCandidates857_0 14024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan857_1

theorem rootBranch857 : search edge 4 [6673] rootCandidates857 6673 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6673] 6673 innerTail857_0 best = best
  exact innerScan857_0

def rootCandidates856 : List Nat := [7687, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates856_checked : candidates edge 5 0 6661 tail857 best.bound = rootCandidates856 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
