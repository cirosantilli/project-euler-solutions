import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates895 : List Nat := [7043, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates895_checked : candidates edge 5 0 6983 tail896 best.bound = rootCandidates895 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates894 : List Nat := [7349, 7577, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates894_checked : candidates edge 5 0 6977 tail895 best.bound = rootCandidates894 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates893 : List Nat := [7211, 7673]
set_option maxRecDepth 4096 in
theorem rootCandidates893_checked : candidates edge 5 0 6971 tail894 best.bound = rootCandidates893 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates892 : List Nat := [7177, 7687]
set_option maxRecDepth 4096 in
theorem rootCandidates892_checked : candidates edge 5 0 6967 tail893 best.bound = rootCandidates892 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates891 : List Nat := [7069, 7243, 7879, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates891_checked : candidates edge 5 0 6961 tail892 best.bound = rootCandidates891 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext q; rfl
  rw [row]
  decide

def innerTail891_4 : List Nat := []
theorem innerScan891_4 : scan edge (search edge 3) 4 [6961] 6961 innerTail891_4 best = best := rfl

def innerTail891_3 : List Nat := 8377 :: innerTail891_4
def innerCandidates891_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates891_3_checked : candidates edge 4 6961 8377 innerTail891_4 best.bound = innerCandidates891_3 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan891_3 : scan edge (search edge 3) 4 [6961] 6961 innerTail891_3 best = best := by
  rw [innerTail891_3, scan, if_neg (by decide), innerCandidates891_3_checked]
  change scan edge (search edge 3) 4 [6961] 6961 innerTail891_4
    (if 3 ≤ innerCandidates891_3.length then search edge 3 [6961, 8377] innerCandidates891_3 15338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan891_4

def innerTail891_2 : List Nat := 7879 :: innerTail891_3
def innerCandidates891_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates891_2_checked : candidates edge 4 6961 7879 innerTail891_3 best.bound = innerCandidates891_2 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan891_2 : scan edge (search edge 3) 4 [6961] 6961 innerTail891_2 best = best := by
  rw [innerTail891_2, scan, if_neg (by decide), innerCandidates891_2_checked]
  change scan edge (search edge 3) 4 [6961] 6961 innerTail891_3
    (if 3 ≤ innerCandidates891_2.length then search edge 3 [6961, 7879] innerCandidates891_2 14840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan891_3

def innerTail891_1 : List Nat := 7243 :: innerTail891_2
def innerCandidates891_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates891_1_checked : candidates edge 4 6961 7243 innerTail891_2 best.bound = innerCandidates891_1 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan891_1 : scan edge (search edge 3) 4 [6961] 6961 innerTail891_1 best = best := by
  rw [innerTail891_1, scan, if_neg (by decide), innerCandidates891_1_checked]
  change scan edge (search edge 3) 4 [6961] 6961 innerTail891_2
    (if 3 ≤ innerCandidates891_1.length then search edge 3 [6961, 7243] innerCandidates891_1 14204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan891_2

def innerTail891_0 : List Nat := 7069 :: innerTail891_1
def innerCandidates891_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates891_0_checked : candidates edge 4 6961 7069 innerTail891_1 best.bound = innerCandidates891_0 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan891_0 : scan edge (search edge 3) 4 [6961] 6961 innerTail891_0 best = best := by
  rw [innerTail891_0, scan, if_neg (by decide), innerCandidates891_0_checked]
  change scan edge (search edge 3) 4 [6961] 6961 innerTail891_1
    (if 3 ≤ innerCandidates891_0.length then search edge 3 [6961, 7069] innerCandidates891_0 14030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan891_1

theorem rootBranch891 : search edge 4 [6961] rootCandidates891 6961 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6961] 6961 innerTail891_0 best = best
  exact innerScan891_0

def rootCandidates890 : List Nat := [7193, 7937, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates890_checked : candidates edge 5 0 6959 tail891 best.bound = rootCandidates890 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates889 : List Nat := [6967, 7591, 7621, 7723, 7741]
set_option maxRecDepth 4096 in
theorem rootCandidates889_checked : candidates edge 5 0 6949 tail890 best.bound = rootCandidates889 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext q; rfl
  rw [row]
  decide

def innerTail889_5 : List Nat := []
theorem innerScan889_5 : scan edge (search edge 3) 4 [6949] 6949 innerTail889_5 best = best := rfl

def innerTail889_4 : List Nat := 7741 :: innerTail889_5
def innerCandidates889_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates889_4_checked : candidates edge 4 6949 7741 innerTail889_5 best.bound = innerCandidates889_4 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan889_4 : scan edge (search edge 3) 4 [6949] 6949 innerTail889_4 best = best := by
  rw [innerTail889_4, scan, if_neg (by decide), innerCandidates889_4_checked]
  change scan edge (search edge 3) 4 [6949] 6949 innerTail889_5
    (if 3 ≤ innerCandidates889_4.length then search edge 3 [6949, 7741] innerCandidates889_4 14690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan889_5

def innerTail889_3 : List Nat := 7723 :: innerTail889_4
def innerCandidates889_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates889_3_checked : candidates edge 4 6949 7723 innerTail889_4 best.bound = innerCandidates889_3 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan889_3 : scan edge (search edge 3) 4 [6949] 6949 innerTail889_3 best = best := by
  rw [innerTail889_3, scan, if_neg (by decide), innerCandidates889_3_checked]
  change scan edge (search edge 3) 4 [6949] 6949 innerTail889_4
    (if 3 ≤ innerCandidates889_3.length then search edge 3 [6949, 7723] innerCandidates889_3 14672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan889_4

def innerTail889_2 : List Nat := 7621 :: innerTail889_3
def innerCandidates889_2 : List Nat := [7741]
set_option maxRecDepth 4096 in
theorem innerCandidates889_2_checked : candidates edge 4 6949 7621 innerTail889_3 best.bound = innerCandidates889_2 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan889_2 : scan edge (search edge 3) 4 [6949] 6949 innerTail889_2 best = best := by
  rw [innerTail889_2, scan, if_neg (by decide), innerCandidates889_2_checked]
  change scan edge (search edge 3) 4 [6949] 6949 innerTail889_3
    (if 3 ≤ innerCandidates889_2.length then search edge 3 [6949, 7621] innerCandidates889_2 14570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan889_3

def innerTail889_1 : List Nat := 7591 :: innerTail889_2
def innerCandidates889_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates889_1_checked : candidates edge 4 6949 7591 innerTail889_2 best.bound = innerCandidates889_1 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan889_1 : scan edge (search edge 3) 4 [6949] 6949 innerTail889_1 best = best := by
  rw [innerTail889_1, scan, if_neg (by decide), innerCandidates889_1_checked]
  change scan edge (search edge 3) 4 [6949] 6949 innerTail889_2
    (if 3 ≤ innerCandidates889_1.length then search edge 3 [6949, 7591] innerCandidates889_1 14540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan889_2

def innerTail889_0 : List Nat := 6967 :: innerTail889_1
def innerCandidates889_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates889_0_checked : candidates edge 4 6949 6967 innerTail889_1 best.bound = innerCandidates889_0 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan889_0 : scan edge (search edge 3) 4 [6949] 6949 innerTail889_0 best = best := by
  rw [innerTail889_0, scan, if_neg (by decide), innerCandidates889_0_checked]
  change scan edge (search edge 3) 4 [6949] 6949 innerTail889_1
    (if 3 ≤ innerCandidates889_0.length then search edge 3 [6949, 6967] innerCandidates889_0 13916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan889_1

theorem rootBranch889 : search edge 4 [6949] rootCandidates889 6949 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6949] 6949 innerTail889_0 best = best
  exact innerScan889_0

def rootCandidates888 : List Nat := [7013, 7829, 8069, 8081, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates888_checked : candidates edge 5 0 6947 tail889 best.bound = rootCandidates888 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext q; rfl
  rw [row]
  decide

def innerTail888_5 : List Nat := []
theorem innerScan888_5 : scan edge (search edge 3) 4 [6947] 6947 innerTail888_5 best = best := rfl

def innerTail888_4 : List Nat := 8369 :: innerTail888_5
def innerCandidates888_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates888_4_checked : candidates edge 4 6947 8369 innerTail888_5 best.bound = innerCandidates888_4 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan888_4 : scan edge (search edge 3) 4 [6947] 6947 innerTail888_4 best = best := by
  rw [innerTail888_4, scan, if_neg (by decide), innerCandidates888_4_checked]
  change scan edge (search edge 3) 4 [6947] 6947 innerTail888_5
    (if 3 ≤ innerCandidates888_4.length then search edge 3 [6947, 8369] innerCandidates888_4 15316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan888_5

def innerTail888_3 : List Nat := 8081 :: innerTail888_4
def innerCandidates888_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates888_3_checked : candidates edge 4 6947 8081 innerTail888_4 best.bound = innerCandidates888_3 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan888_3 : scan edge (search edge 3) 4 [6947] 6947 innerTail888_3 best = best := by
  rw [innerTail888_3, scan, if_neg (by decide), innerCandidates888_3_checked]
  change scan edge (search edge 3) 4 [6947] 6947 innerTail888_4
    (if 3 ≤ innerCandidates888_3.length then search edge 3 [6947, 8081] innerCandidates888_3 15028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan888_4

def innerTail888_2 : List Nat := 8069 :: innerTail888_3
def innerCandidates888_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates888_2_checked : candidates edge 4 6947 8069 innerTail888_3 best.bound = innerCandidates888_2 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan888_2 : scan edge (search edge 3) 4 [6947] 6947 innerTail888_2 best = best := by
  rw [innerTail888_2, scan, if_neg (by decide), innerCandidates888_2_checked]
  change scan edge (search edge 3) 4 [6947] 6947 innerTail888_3
    (if 3 ≤ innerCandidates888_2.length then search edge 3 [6947, 8069] innerCandidates888_2 15016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan888_3

def innerTail888_1 : List Nat := 7829 :: innerTail888_2
def innerCandidates888_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates888_1_checked : candidates edge 4 6947 7829 innerTail888_2 best.bound = innerCandidates888_1 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan888_1 : scan edge (search edge 3) 4 [6947] 6947 innerTail888_1 best = best := by
  rw [innerTail888_1, scan, if_neg (by decide), innerCandidates888_1_checked]
  change scan edge (search edge 3) 4 [6947] 6947 innerTail888_2
    (if 3 ≤ innerCandidates888_1.length then search edge 3 [6947, 7829] innerCandidates888_1 14776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan888_2

def innerTail888_0 : List Nat := 7013 :: innerTail888_1
def innerCandidates888_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates888_0_checked : candidates edge 4 6947 7013 innerTail888_1 best.bound = innerCandidates888_0 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan888_0 : scan edge (search edge 3) 4 [6947] 6947 innerTail888_0 best = best := by
  rw [innerTail888_0, scan, if_neg (by decide), innerCandidates888_0_checked]
  change scan edge (search edge 3) 4 [6947] 6947 innerTail888_1
    (if 3 ≤ innerCandidates888_0.length then search edge 3 [6947, 7013] innerCandidates888_0 13960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan888_1

theorem rootBranch888 : search edge 4 [6947] rootCandidates888 6947 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6947] 6947 innerTail888_0 best = best
  exact innerScan888_0

end ProjectEulerSolutions.PrimePairUpper.Search
