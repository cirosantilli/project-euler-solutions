import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates4 : List Nat := [83, 239, 257, 383, 389, 431, 449, 491, 881, 971, 1049, 1553, 2027, 2069, 2153, 2213, 2243, 2297, 2351, 2411, 2657, 2741, 3023, 3137, 3191, 3347, 3359, 3539, 3917, 4019, 4079, 4157, 4241, 4583, 4637, 4649, 4673, 4703, 4877, 4943, 5003, 5261, 5333, 5519, 5783, 5897, 6317, 6353, 6521, 6599, 6857, 6899, 7109, 7127, 7211, 7283, 7487, 7691, 7841, 7883, 8069, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates4_checked : candidates edge 5 0 17 tail5 best.bound = rootCandidates4 := by
  unfold candidates
  have row : edge 17 = PrimePairCache.smallRow17.contains := by funext q; rfl
  rw [row]
  decide

def innerTail4_62 : List Nat := []
theorem innerScan4_62 : scan edge (search edge 3) 4 [17] 17 innerTail4_62 best = best := rfl

def innerTail4_61 : List Nat := 8231 :: innerTail4_62
def innerCandidates4_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_61_checked : candidates edge 4 17 8231 innerTail4_62 best.bound = innerCandidates4_61 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_61 : scan edge (search edge 3) 4 [17] 17 innerTail4_61 best = best := by
  rw [innerTail4_61, scan, if_neg (by decide), innerCandidates4_61_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_62
    (if 3 ≤ innerCandidates4_61.length then search edge 3 [17, 8231] innerCandidates4_61 8248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_62

def innerTail4_60 : List Nat := 8069 :: innerTail4_61
def innerCandidates4_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_60_checked : candidates edge 4 17 8069 innerTail4_61 best.bound = innerCandidates4_60 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_60 : scan edge (search edge 3) 4 [17] 17 innerTail4_60 best = best := by
  rw [innerTail4_60, scan, if_neg (by decide), innerCandidates4_60_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_61
    (if 3 ≤ innerCandidates4_60.length then search edge 3 [17, 8069] innerCandidates4_60 8086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_61

def innerTail4_59 : List Nat := 7883 :: innerTail4_60
def innerCandidates4_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_59_checked : candidates edge 4 17 7883 innerTail4_60 best.bound = innerCandidates4_59 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_59 : scan edge (search edge 3) 4 [17] 17 innerTail4_59 best = best := by
  rw [innerTail4_59, scan, if_neg (by decide), innerCandidates4_59_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_60
    (if 3 ≤ innerCandidates4_59.length then search edge 3 [17, 7883] innerCandidates4_59 7900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_60

def innerTail4_58 : List Nat := 7841 :: innerTail4_59
def innerCandidates4_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_58_checked : candidates edge 4 17 7841 innerTail4_59 best.bound = innerCandidates4_58 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_58 : scan edge (search edge 3) 4 [17] 17 innerTail4_58 best = best := by
  rw [innerTail4_58, scan, if_neg (by decide), innerCandidates4_58_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_59
    (if 3 ≤ innerCandidates4_58.length then search edge 3 [17, 7841] innerCandidates4_58 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_59

def innerTail4_57 : List Nat := 7691 :: innerTail4_58
def innerCandidates4_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_57_checked : candidates edge 4 17 7691 innerTail4_58 best.bound = innerCandidates4_57 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_57 : scan edge (search edge 3) 4 [17] 17 innerTail4_57 best = best := by
  rw [innerTail4_57, scan, if_neg (by decide), innerCandidates4_57_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_58
    (if 3 ≤ innerCandidates4_57.length then search edge 3 [17, 7691] innerCandidates4_57 7708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_58

def innerTail4_56 : List Nat := 7487 :: innerTail4_57
def innerCandidates4_56 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates4_56_checked : candidates edge 4 17 7487 innerTail4_57 best.bound = innerCandidates4_56 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_56 : scan edge (search edge 3) 4 [17] 17 innerTail4_56 best = best := by
  rw [innerTail4_56, scan, if_neg (by decide), innerCandidates4_56_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_57
    (if 3 ≤ innerCandidates4_56.length then search edge 3 [17, 7487] innerCandidates4_56 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_57

def innerTail4_55 : List Nat := 7283 :: innerTail4_56
def innerCandidates4_55 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_55_checked : candidates edge 4 17 7283 innerTail4_56 best.bound = innerCandidates4_55 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_55 : scan edge (search edge 3) 4 [17] 17 innerTail4_55 best = best := by
  rw [innerTail4_55, scan, if_neg (by decide), innerCandidates4_55_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_56
    (if 3 ≤ innerCandidates4_55.length then search edge 3 [17, 7283] innerCandidates4_55 7300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_56

def innerTail4_54 : List Nat := 7211 :: innerTail4_55
def innerCandidates4_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_54_checked : candidates edge 4 17 7211 innerTail4_55 best.bound = innerCandidates4_54 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_54 : scan edge (search edge 3) 4 [17] 17 innerTail4_54 best = best := by
  rw [innerTail4_54, scan, if_neg (by decide), innerCandidates4_54_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_55
    (if 3 ≤ innerCandidates4_54.length then search edge 3 [17, 7211] innerCandidates4_54 7228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_55

def innerTail4_53 : List Nat := 7127 :: innerTail4_54
def innerCandidates4_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_53_checked : candidates edge 4 17 7127 innerTail4_54 best.bound = innerCandidates4_53 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_53 : scan edge (search edge 3) 4 [17] 17 innerTail4_53 best = best := by
  rw [innerTail4_53, scan, if_neg (by decide), innerCandidates4_53_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_54
    (if 3 ≤ innerCandidates4_53.length then search edge 3 [17, 7127] innerCandidates4_53 7144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_54

def innerTail4_52 : List Nat := 7109 :: innerTail4_53
def innerCandidates4_52 : List Nat := [7127, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates4_52_checked : candidates edge 4 17 7109 innerTail4_53 best.bound = innerCandidates4_52 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_52 : scan edge (search edge 3) 4 [17] 17 innerTail4_52 best = best := by
  rw [innerTail4_52, scan, if_neg (by decide), innerCandidates4_52_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_53
    (if 3 ≤ innerCandidates4_52.length then search edge 3 [17, 7109] innerCandidates4_52 7126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_53

def innerTail4_51 : List Nat := 6899 :: innerTail4_52
def innerCandidates4_51 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates4_51_checked : candidates edge 4 17 6899 innerTail4_52 best.bound = innerCandidates4_51 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_51 : scan edge (search edge 3) 4 [17] 17 innerTail4_51 best = best := by
  rw [innerTail4_51, scan, if_neg (by decide), innerCandidates4_51_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_52
    (if 3 ≤ innerCandidates4_51.length then search edge 3 [17, 6899] innerCandidates4_51 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_52

def innerTail4_50 : List Nat := 6857 :: innerTail4_51
def innerCandidates4_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_50_checked : candidates edge 4 17 6857 innerTail4_51 best.bound = innerCandidates4_50 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_50 : scan edge (search edge 3) 4 [17] 17 innerTail4_50 best = best := by
  rw [innerTail4_50, scan, if_neg (by decide), innerCandidates4_50_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_51
    (if 3 ≤ innerCandidates4_50.length then search edge 3 [17, 6857] innerCandidates4_50 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_51

def innerTail4_49 : List Nat := 6599 :: innerTail4_50
def innerCandidates4_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_49_checked : candidates edge 4 17 6599 innerTail4_50 best.bound = innerCandidates4_49 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_49 : scan edge (search edge 3) 4 [17] 17 innerTail4_49 best = best := by
  rw [innerTail4_49, scan, if_neg (by decide), innerCandidates4_49_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_50
    (if 3 ≤ innerCandidates4_49.length then search edge 3 [17, 6599] innerCandidates4_49 6616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_50

def innerTail4_48 : List Nat := 6521 :: innerTail4_49
def innerCandidates4_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_48_checked : candidates edge 4 17 6521 innerTail4_49 best.bound = innerCandidates4_48 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_48 : scan edge (search edge 3) 4 [17] 17 innerTail4_48 best = best := by
  rw [innerTail4_48, scan, if_neg (by decide), innerCandidates4_48_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_49
    (if 3 ≤ innerCandidates4_48.length then search edge 3 [17, 6521] innerCandidates4_48 6538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_49

def innerTail4_47 : List Nat := 6353 :: innerTail4_48
def innerCandidates4_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_47_checked : candidates edge 4 17 6353 innerTail4_48 best.bound = innerCandidates4_47 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_47 : scan edge (search edge 3) 4 [17] 17 innerTail4_47 best = best := by
  rw [innerTail4_47, scan, if_neg (by decide), innerCandidates4_47_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_48
    (if 3 ≤ innerCandidates4_47.length then search edge 3 [17, 6353] innerCandidates4_47 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_48

def innerTail4_46 : List Nat := 6317 :: innerTail4_47
def innerCandidates4_46 : List Nat := [6599]
set_option maxRecDepth 4096 in
theorem innerCandidates4_46_checked : candidates edge 4 17 6317 innerTail4_47 best.bound = innerCandidates4_46 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_46 : scan edge (search edge 3) 4 [17] 17 innerTail4_46 best = best := by
  rw [innerTail4_46, scan, if_neg (by decide), innerCandidates4_46_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_47
    (if 3 ≤ innerCandidates4_46.length then search edge 3 [17, 6317] innerCandidates4_46 6334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_47

def innerTail4_45 : List Nat := 5897 :: innerTail4_46
def innerCandidates4_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_45_checked : candidates edge 4 17 5897 innerTail4_46 best.bound = innerCandidates4_45 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_45 : scan edge (search edge 3) 4 [17] 17 innerTail4_45 best = best := by
  rw [innerTail4_45, scan, if_neg (by decide), innerCandidates4_45_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_46
    (if 3 ≤ innerCandidates4_45.length then search edge 3 [17, 5897] innerCandidates4_45 5914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_46

def innerTail4_44 : List Nat := 5783 :: innerTail4_45
def innerCandidates4_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_44_checked : candidates edge 4 17 5783 innerTail4_45 best.bound = innerCandidates4_44 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_44 : scan edge (search edge 3) 4 [17] 17 innerTail4_44 best = best := by
  rw [innerTail4_44, scan, if_neg (by decide), innerCandidates4_44_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_45
    (if 3 ≤ innerCandidates4_44.length then search edge 3 [17, 5783] innerCandidates4_44 5800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_45

def innerTail4_43 : List Nat := 5519 :: innerTail4_44
def innerCandidates4_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_43_checked : candidates edge 4 17 5519 innerTail4_44 best.bound = innerCandidates4_43 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_43 : scan edge (search edge 3) 4 [17] 17 innerTail4_43 best = best := by
  rw [innerTail4_43, scan, if_neg (by decide), innerCandidates4_43_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_44
    (if 3 ≤ innerCandidates4_43.length then search edge 3 [17, 5519] innerCandidates4_43 5536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_44

def innerTail4_42 : List Nat := 5333 :: innerTail4_43
def innerCandidates4_42 : List Nat := [7283]
set_option maxRecDepth 4096 in
theorem innerCandidates4_42_checked : candidates edge 4 17 5333 innerTail4_43 best.bound = innerCandidates4_42 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_42 : scan edge (search edge 3) 4 [17] 17 innerTail4_42 best = best := by
  rw [innerTail4_42, scan, if_neg (by decide), innerCandidates4_42_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_43
    (if 3 ≤ innerCandidates4_42.length then search edge 3 [17, 5333] innerCandidates4_42 5350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_43

def innerTail4_41 : List Nat := 5261 :: innerTail4_42
def innerCandidates4_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_41_checked : candidates edge 4 17 5261 innerTail4_42 best.bound = innerCandidates4_41 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_41 : scan edge (search edge 3) 4 [17] 17 innerTail4_41 best = best := by
  rw [innerTail4_41, scan, if_neg (by decide), innerCandidates4_41_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_42
    (if 3 ≤ innerCandidates4_41.length then search edge 3 [17, 5261] innerCandidates4_41 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_42

def innerTail4_40 : List Nat := 5003 :: innerTail4_41
def innerCandidates4_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_40_checked : candidates edge 4 17 5003 innerTail4_41 best.bound = innerCandidates4_40 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_40 : scan edge (search edge 3) 4 [17] 17 innerTail4_40 best = best := by
  rw [innerTail4_40, scan, if_neg (by decide), innerCandidates4_40_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_41
    (if 3 ≤ innerCandidates4_40.length then search edge 3 [17, 5003] innerCandidates4_40 5020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_41

def innerTail4_39 : List Nat := 4943 :: innerTail4_40
def innerCandidates4_39 : List Nat := [5261]
set_option maxRecDepth 4096 in
theorem innerCandidates4_39_checked : candidates edge 4 17 4943 innerTail4_40 best.bound = innerCandidates4_39 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_39 : scan edge (search edge 3) 4 [17] 17 innerTail4_39 best = best := by
  rw [innerTail4_39, scan, if_neg (by decide), innerCandidates4_39_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_40
    (if 3 ≤ innerCandidates4_39.length then search edge 3 [17, 4943] innerCandidates4_39 4960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_40

def innerTail4_38 : List Nat := 4877 :: innerTail4_39
def innerCandidates4_38 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates4_38_checked : candidates edge 4 17 4877 innerTail4_39 best.bound = innerCandidates4_38 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_38 : scan edge (search edge 3) 4 [17] 17 innerTail4_38 best = best := by
  rw [innerTail4_38, scan, if_neg (by decide), innerCandidates4_38_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_39
    (if 3 ≤ innerCandidates4_38.length then search edge 3 [17, 4877] innerCandidates4_38 4894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_39

def innerTail4_37 : List Nat := 4703 :: innerTail4_38
def innerCandidates4_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_37_checked : candidates edge 4 17 4703 innerTail4_38 best.bound = innerCandidates4_37 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_37 : scan edge (search edge 3) 4 [17] 17 innerTail4_37 best = best := by
  rw [innerTail4_37, scan, if_neg (by decide), innerCandidates4_37_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_38
    (if 3 ≤ innerCandidates4_37.length then search edge 3 [17, 4703] innerCandidates4_37 4720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_38

def innerTail4_36 : List Nat := 4673 :: innerTail4_37
def innerCandidates4_36 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates4_36_checked : candidates edge 4 17 4673 innerTail4_37 best.bound = innerCandidates4_36 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_36 : scan edge (search edge 3) 4 [17] 17 innerTail4_36 best = best := by
  rw [innerTail4_36, scan, if_neg (by decide), innerCandidates4_36_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_37
    (if 3 ≤ innerCandidates4_36.length then search edge 3 [17, 4673] innerCandidates4_36 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_37

def innerTail4_35 : List Nat := 4649 :: innerTail4_36
def innerCandidates4_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_35_checked : candidates edge 4 17 4649 innerTail4_36 best.bound = innerCandidates4_35 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_35 : scan edge (search edge 3) 4 [17] 17 innerTail4_35 best = best := by
  rw [innerTail4_35, scan, if_neg (by decide), innerCandidates4_35_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_36
    (if 3 ≤ innerCandidates4_35.length then search edge 3 [17, 4649] innerCandidates4_35 4666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_36

def innerTail4_34 : List Nat := 4637 :: innerTail4_35
def innerCandidates4_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_34_checked : candidates edge 4 17 4637 innerTail4_35 best.bound = innerCandidates4_34 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_34 : scan edge (search edge 3) 4 [17] 17 innerTail4_34 best = best := by
  rw [innerTail4_34, scan, if_neg (by decide), innerCandidates4_34_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_35
    (if 3 ≤ innerCandidates4_34.length then search edge 3 [17, 4637] innerCandidates4_34 4654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_35

def innerTail4_33 : List Nat := 4583 :: innerTail4_34
def innerCandidates4_33 : List Nat := [4637, 5003]
set_option maxRecDepth 4096 in
theorem innerCandidates4_33_checked : candidates edge 4 17 4583 innerTail4_34 best.bound = innerCandidates4_33 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_33 : scan edge (search edge 3) 4 [17] 17 innerTail4_33 best = best := by
  rw [innerTail4_33, scan, if_neg (by decide), innerCandidates4_33_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_34
    (if 3 ≤ innerCandidates4_33.length then search edge 3 [17, 4583] innerCandidates4_33 4600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_34

def innerTail4_32 : List Nat := 4241 :: innerTail4_33
def innerCandidates4_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_32_checked : candidates edge 4 17 4241 innerTail4_33 best.bound = innerCandidates4_32 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_32 : scan edge (search edge 3) 4 [17] 17 innerTail4_32 best = best := by
  rw [innerTail4_32, scan, if_neg (by decide), innerCandidates4_32_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_33
    (if 3 ≤ innerCandidates4_32.length then search edge 3 [17, 4241] innerCandidates4_32 4258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_33

def innerTail4_31 : List Nat := 4157 :: innerTail4_32
def innerCandidates4_31 : List Nat := [5333, 7127]
set_option maxRecDepth 4096 in
theorem innerCandidates4_31_checked : candidates edge 4 17 4157 innerTail4_32 best.bound = innerCandidates4_31 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_31 : scan edge (search edge 3) 4 [17] 17 innerTail4_31 best = best := by
  rw [innerTail4_31, scan, if_neg (by decide), innerCandidates4_31_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_32
    (if 3 ≤ innerCandidates4_31.length then search edge 3 [17, 4157] innerCandidates4_31 4174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_32

def innerTail4_30 : List Nat := 4079 :: innerTail4_31
def innerCandidates4_30 : List Nat := [5897]
set_option maxRecDepth 4096 in
theorem innerCandidates4_30_checked : candidates edge 4 17 4079 innerTail4_31 best.bound = innerCandidates4_30 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_30 : scan edge (search edge 3) 4 [17] 17 innerTail4_30 best = best := by
  rw [innerTail4_30, scan, if_neg (by decide), innerCandidates4_30_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_31
    (if 3 ≤ innerCandidates4_30.length then search edge 3 [17, 4079] innerCandidates4_30 4096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_31

def innerTail4_29 : List Nat := 4019 :: innerTail4_30
def innerCandidates4_29 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates4_29_checked : candidates edge 4 17 4019 innerTail4_30 best.bound = innerCandidates4_29 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_29 : scan edge (search edge 3) 4 [17] 17 innerTail4_29 best = best := by
  rw [innerTail4_29, scan, if_neg (by decide), innerCandidates4_29_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_30
    (if 3 ≤ innerCandidates4_29.length then search edge 3 [17, 4019] innerCandidates4_29 4036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_30

def innerTail4_28 : List Nat := 3917 :: innerTail4_29
def innerCandidates4_28 : List Nat := [4019, 4649, 6899]
set_option maxRecDepth 4096 in
theorem innerCandidates4_28_checked : candidates edge 4 17 3917 innerTail4_29 best.bound = innerCandidates4_28 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_28 : search edge 3 [17, 3917] innerCandidates4_28 3934 best = best := by decide

theorem innerScan4_28 : scan edge (search edge 3) 4 [17] 17 innerTail4_28 best = best := by
  rw [innerTail4_28, scan, if_neg (by decide), innerCandidates4_28_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_29
    (if 3 ≤ innerCandidates4_28.length then search edge 3 [17, 3917] innerCandidates4_28 3934 best else best) = best
  rw [if_pos (by decide), innerChild4_28]
  exact innerScan4_29

def innerTail4_27 : List Nat := 3539 :: innerTail4_28
def innerCandidates4_27 : List Nat := [4649, 6521]
set_option maxRecDepth 4096 in
theorem innerCandidates4_27_checked : candidates edge 4 17 3539 innerTail4_28 best.bound = innerCandidates4_27 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_27 : scan edge (search edge 3) 4 [17] 17 innerTail4_27 best = best := by
  rw [innerTail4_27, scan, if_neg (by decide), innerCandidates4_27_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_28
    (if 3 ≤ innerCandidates4_27.length then search edge 3 [17, 3539] innerCandidates4_27 3556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_28

def innerTail4_26 : List Nat := 3359 :: innerTail4_27
def innerCandidates4_26 : List Nat := [6599]
set_option maxRecDepth 4096 in
theorem innerCandidates4_26_checked : candidates edge 4 17 3359 innerTail4_27 best.bound = innerCandidates4_26 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_26 : scan edge (search edge 3) 4 [17] 17 innerTail4_26 best = best := by
  rw [innerTail4_26, scan, if_neg (by decide), innerCandidates4_26_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_27
    (if 3 ≤ innerCandidates4_26.length then search edge 3 [17, 3359] innerCandidates4_26 3376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_27

def innerTail4_25 : List Nat := 3347 :: innerTail4_26
def innerCandidates4_25 : List Nat := [4673, 4703, 5333, 8231]
set_option maxRecDepth 4096 in
theorem innerCandidates4_25_checked : candidates edge 4 17 3347 innerTail4_26 best.bound = innerCandidates4_25 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_25 : search edge 3 [17, 3347] innerCandidates4_25 3364 best = best := by decide

theorem innerScan4_25 : scan edge (search edge 3) 4 [17] 17 innerTail4_25 best = best := by
  rw [innerTail4_25, scan, if_neg (by decide), innerCandidates4_25_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_26
    (if 3 ≤ innerCandidates4_25.length then search edge 3 [17, 3347] innerCandidates4_25 3364 best else best) = best
  rw [if_pos (by decide), innerChild4_25]
  exact innerScan4_26

def innerTail4_24 : List Nat := 3191 :: innerTail4_25
def innerCandidates4_24 : List Nat := [4583, 7883, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates4_24_checked : candidates edge 4 17 3191 innerTail4_25 best.bound = innerCandidates4_24 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_24 : search edge 3 [17, 3191] innerCandidates4_24 3208 best = best := by decide

theorem innerScan4_24 : scan edge (search edge 3) 4 [17] 17 innerTail4_24 best = best := by
  rw [innerTail4_24, scan, if_neg (by decide), innerCandidates4_24_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_25
    (if 3 ≤ innerCandidates4_24.length then search edge 3 [17, 3191] innerCandidates4_24 3208 best else best) = best
  rw [if_pos (by decide), innerChild4_24]
  exact innerScan4_25

def innerTail4_23 : List Nat := 3137 :: innerTail4_24
def innerCandidates4_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_23_checked : candidates edge 4 17 3137 innerTail4_24 best.bound = innerCandidates4_23 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_23 : scan edge (search edge 3) 4 [17] 17 innerTail4_23 best = best := by
  rw [innerTail4_23, scan, if_neg (by decide), innerCandidates4_23_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_24
    (if 3 ≤ innerCandidates4_23.length then search edge 3 [17, 3137] innerCandidates4_23 3154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_24

def innerTail4_22 : List Nat := 3023 :: innerTail4_23
def innerCandidates4_22 : List Nat := [4637, 4649, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates4_22_checked : candidates edge 4 17 3023 innerTail4_23 best.bound = innerCandidates4_22 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_22 : search edge 3 [17, 3023] innerCandidates4_22 3040 best = best := by decide

theorem innerScan4_22 : scan edge (search edge 3) 4 [17] 17 innerTail4_22 best = best := by
  rw [innerTail4_22, scan, if_neg (by decide), innerCandidates4_22_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_23
    (if 3 ≤ innerCandidates4_22.length then search edge 3 [17, 3023] innerCandidates4_22 3040 best else best) = best
  rw [if_pos (by decide), innerChild4_22]
  exact innerScan4_23

def innerTail4_21 : List Nat := 2741 :: innerTail4_22
def innerCandidates4_21 : List Nat := [3917, 4649, 6353, 6899]
set_option maxRecDepth 4096 in
theorem innerCandidates4_21_checked : candidates edge 4 17 2741 innerTail4_22 best.bound = innerCandidates4_21 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_21 : search edge 3 [17, 2741] innerCandidates4_21 2758 best = best := by decide

theorem innerScan4_21 : scan edge (search edge 3) 4 [17] 17 innerTail4_21 best = best := by
  rw [innerTail4_21, scan, if_neg (by decide), innerCandidates4_21_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_22
    (if 3 ≤ innerCandidates4_21.length then search edge 3 [17, 2741] innerCandidates4_21 2758 best else best) = best
  rw [if_pos (by decide), innerChild4_21]
  exact innerScan4_22

def innerTail4_20 : List Nat := 2657 :: innerTail4_21
def innerCandidates4_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_20_checked : candidates edge 4 17 2657 innerTail4_21 best.bound = innerCandidates4_20 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_20 : scan edge (search edge 3) 4 [17] 17 innerTail4_20 best = best := by
  rw [innerTail4_20, scan, if_neg (by decide), innerCandidates4_20_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_21
    (if 3 ≤ innerCandidates4_20.length then search edge 3 [17, 2657] innerCandidates4_20 2674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_21

def innerTail4_19 : List Nat := 2411 :: innerTail4_20
def innerCandidates4_19 : List Nat := [3917, 4241, 5519]
set_option maxRecDepth 4096 in
theorem innerCandidates4_19_checked : candidates edge 4 17 2411 innerTail4_20 best.bound = innerCandidates4_19 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_19 : search edge 3 [17, 2411] innerCandidates4_19 2428 best = best := by decide

theorem innerScan4_19 : scan edge (search edge 3) 4 [17] 17 innerTail4_19 best = best := by
  rw [innerTail4_19, scan, if_neg (by decide), innerCandidates4_19_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_20
    (if 3 ≤ innerCandidates4_19.length then search edge 3 [17, 2411] innerCandidates4_19 2428 best else best) = best
  rw [if_pos (by decide), innerChild4_19]
  exact innerScan4_20

def innerTail4_18 : List Nat := 2351 :: innerTail4_19
def innerCandidates4_18 : List Nat := [5519, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates4_18_checked : candidates edge 4 17 2351 innerTail4_19 best.bound = innerCandidates4_18 := by
  unfold candidates
  have row : edge 2351 = PrimePairCache.smallRow2351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_18 : scan edge (search edge 3) 4 [17] 17 innerTail4_18 best = best := by
  rw [innerTail4_18, scan, if_neg (by decide), innerCandidates4_18_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_19
    (if 3 ≤ innerCandidates4_18.length then search edge 3 [17, 2351] innerCandidates4_18 2368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_19

def innerTail4_17 : List Nat := 2297 :: innerTail4_18
def innerCandidates4_17 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates4_17_checked : candidates edge 4 17 2297 innerTail4_18 best.bound = innerCandidates4_17 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_17 : scan edge (search edge 3) 4 [17] 17 innerTail4_17 best = best := by
  rw [innerTail4_17, scan, if_neg (by decide), innerCandidates4_17_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_18
    (if 3 ≤ innerCandidates4_17.length then search edge 3 [17, 2297] innerCandidates4_17 2314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_18

def innerTail4_16 : List Nat := 2243 :: innerTail4_17
def innerCandidates4_16 : List Nat := [2411, 3539, 6599, 6899, 7487]
set_option maxRecDepth 4096 in
theorem innerCandidates4_16_checked : candidates edge 4 17 2243 innerTail4_17 best.bound = innerCandidates4_16 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_16 : search edge 3 [17, 2243] innerCandidates4_16 2260 best = best := by decide

theorem innerScan4_16 : scan edge (search edge 3) 4 [17] 17 innerTail4_16 best = best := by
  rw [innerTail4_16, scan, if_neg (by decide), innerCandidates4_16_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_17
    (if 3 ≤ innerCandidates4_16.length then search edge 3 [17, 2243] innerCandidates4_16 2260 best else best) = best
  rw [if_pos (by decide), innerChild4_16]
  exact innerScan4_17

def innerTail4_15 : List Nat := 2213 :: innerTail4_16
def innerCandidates4_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_15_checked : candidates edge 4 17 2213 innerTail4_16 best.bound = innerCandidates4_15 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_15 : scan edge (search edge 3) 4 [17] 17 innerTail4_15 best = best := by
  rw [innerTail4_15, scan, if_neg (by decide), innerCandidates4_15_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_16
    (if 3 ≤ innerCandidates4_15.length then search edge 3 [17, 2213] innerCandidates4_15 2230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_16

def innerTail4_14 : List Nat := 2153 :: innerTail4_15
def innerCandidates4_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_14_checked : candidates edge 4 17 2153 innerTail4_15 best.bound = innerCandidates4_14 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_14 : scan edge (search edge 3) 4 [17] 17 innerTail4_14 best = best := by
  rw [innerTail4_14, scan, if_neg (by decide), innerCandidates4_14_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_15
    (if 3 ≤ innerCandidates4_14.length then search edge 3 [17, 2153] innerCandidates4_14 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_15

def innerTail4_13 : List Nat := 2069 :: innerTail4_14
def innerCandidates4_13 : List Nat := [2297, 4637, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates4_13_checked : candidates edge 4 17 2069 innerTail4_14 best.bound = innerCandidates4_13 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_13 : search edge 3 [17, 2069] innerCandidates4_13 2086 best = best := by decide

theorem innerScan4_13 : scan edge (search edge 3) 4 [17] 17 innerTail4_13 best = best := by
  rw [innerTail4_13, scan, if_neg (by decide), innerCandidates4_13_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_14
    (if 3 ≤ innerCandidates4_13.length then search edge 3 [17, 2069] innerCandidates4_13 2086 best else best) = best
  rw [if_pos (by decide), innerChild4_13]
  exact innerScan4_14

def innerTail4_12 : List Nat := 2027 :: innerTail4_13
def innerCandidates4_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_12_checked : candidates edge 4 17 2027 innerTail4_13 best.bound = innerCandidates4_12 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_12 : scan edge (search edge 3) 4 [17] 17 innerTail4_12 best = best := by
  rw [innerTail4_12, scan, if_neg (by decide), innerCandidates4_12_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_13
    (if 3 ≤ innerCandidates4_12.length then search edge 3 [17, 2027] innerCandidates4_12 2044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_13

def innerTail4_11 : List Nat := 1553 :: innerTail4_12
def innerCandidates4_11 : List Nat := [7211, 7883]
set_option maxRecDepth 4096 in
theorem innerCandidates4_11_checked : candidates edge 4 17 1553 innerTail4_12 best.bound = innerCandidates4_11 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_11 : scan edge (search edge 3) 4 [17] 17 innerTail4_11 best = best := by
  rw [innerTail4_11, scan, if_neg (by decide), innerCandidates4_11_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_12
    (if 3 ≤ innerCandidates4_11.length then search edge 3 [17, 1553] innerCandidates4_11 1570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_12

def innerTail4_10 : List Nat := 1049 :: innerTail4_11
def innerCandidates4_10 : List Nat := [2297, 5897, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates4_10_checked : candidates edge 4 17 1049 innerTail4_11 best.bound = innerCandidates4_10 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_10 : search edge 3 [17, 1049] innerCandidates4_10 1066 best = best := by decide

theorem innerScan4_10 : scan edge (search edge 3) 4 [17] 17 innerTail4_10 best = best := by
  rw [innerTail4_10, scan, if_neg (by decide), innerCandidates4_10_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_11
    (if 3 ≤ innerCandidates4_10.length then search edge 3 [17, 1049] innerCandidates4_10 1066 best else best) = best
  rw [if_pos (by decide), innerChild4_10]
  exact innerScan4_11

def innerTail4_9 : List Nat := 971 :: innerTail4_10
def innerCandidates4_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates4_9_checked : candidates edge 4 17 971 innerTail4_10 best.bound = innerCandidates4_9 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_9 : scan edge (search edge 3) 4 [17] 17 innerTail4_9 best = best := by
  rw [innerTail4_9, scan, if_neg (by decide), innerCandidates4_9_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_10
    (if 3 ≤ innerCandidates4_9.length then search edge 3 [17, 971] innerCandidates4_9 988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_10

def innerTail4_8 : List Nat := 881 :: innerTail4_9
def innerCandidates4_8 : List Nat := [2411]
set_option maxRecDepth 4096 in
theorem innerCandidates4_8_checked : candidates edge 4 17 881 innerTail4_9 best.bound = innerCandidates4_8 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_8 : scan edge (search edge 3) 4 [17] 17 innerTail4_8 best = best := by
  rw [innerTail4_8, scan, if_neg (by decide), innerCandidates4_8_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_9
    (if 3 ≤ innerCandidates4_8.length then search edge 3 [17, 881] innerCandidates4_8 898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_9

def innerTail4_7 : List Nat := 491 :: innerTail4_8
def innerCandidates4_7 : List Nat := [3917]
set_option maxRecDepth 4096 in
theorem innerCandidates4_7_checked : candidates edge 4 17 491 innerTail4_8 best.bound = innerCandidates4_7 := by
  unfold candidates
  have row : edge 491 = PrimePairCache.smallRow491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_7 : scan edge (search edge 3) 4 [17] 17 innerTail4_7 best = best := by
  rw [innerTail4_7, scan, if_neg (by decide), innerCandidates4_7_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_8
    (if 3 ≤ innerCandidates4_7.length then search edge 3 [17, 491] innerCandidates4_7 508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_8

def innerTail4_6 : List Nat := 449 :: innerTail4_7
def innerCandidates4_6 : List Nat := [2069, 4157, 6353, 6599, 7109]
set_option maxRecDepth 4096 in
theorem innerCandidates4_6_checked : candidates edge 4 17 449 innerTail4_7 best.bound = innerCandidates4_6 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_6 : search edge 3 [17, 449] innerCandidates4_6 466 best = best := by decide

theorem innerScan4_6 : scan edge (search edge 3) 4 [17] 17 innerTail4_6 best = best := by
  rw [innerTail4_6, scan, if_neg (by decide), innerCandidates4_6_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_7
    (if 3 ≤ innerCandidates4_6.length then search edge 3 [17, 449] innerCandidates4_6 466 best else best) = best
  rw [if_pos (by decide), innerChild4_6]
  exact innerScan4_7

def innerTail4_5 : List Nat := 431 :: innerTail4_6
def innerCandidates4_5 : List Nat := [4943, 7127, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates4_5_checked : candidates edge 4 17 431 innerTail4_6 best.bound = innerCandidates4_5 := by
  unfold candidates
  have row : edge 431 = PrimePairCache.smallRow431.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_5 : search edge 3 [17, 431] innerCandidates4_5 448 best = best := by decide

theorem innerScan4_5 : scan edge (search edge 3) 4 [17] 17 innerTail4_5 best = best := by
  rw [innerTail4_5, scan, if_neg (by decide), innerCandidates4_5_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_6
    (if 3 ≤ innerCandidates4_5.length then search edge 3 [17, 431] innerCandidates4_5 448 best else best) = best
  rw [if_pos (by decide), innerChild4_5]
  exact innerScan4_6

def innerTail4_4 : List Nat := 389 :: innerTail4_5
def innerCandidates4_4 : List Nat := [971, 3137, 3191, 4637, 4703]
set_option maxRecDepth 4096 in
theorem innerCandidates4_4_checked : candidates edge 4 17 389 innerTail4_5 best.bound = innerCandidates4_4 := by
  unfold candidates
  have row : edge 389 = PrimePairCache.smallRow389.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_4 : search edge 3 [17, 389] innerCandidates4_4 406 best = best := by decide

theorem innerScan4_4 : scan edge (search edge 3) 4 [17] 17 innerTail4_4 best = best := by
  rw [innerTail4_4, scan, if_neg (by decide), innerCandidates4_4_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_5
    (if 3 ≤ innerCandidates4_4.length then search edge 3 [17, 389] innerCandidates4_4 406 best else best) = best
  rw [if_pos (by decide), innerChild4_4]
  exact innerScan4_5

def innerTail4_3 : List Nat := 383 :: innerTail4_4
def innerCandidates4_3 : List Nat := [3359, 6317]
set_option maxRecDepth 4096 in
theorem innerCandidates4_3_checked : candidates edge 4 17 383 innerTail4_4 best.bound = innerCandidates4_3 := by
  unfold candidates
  have row : edge 383 = PrimePairCache.smallRow383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan4_3 : scan edge (search edge 3) 4 [17] 17 innerTail4_3 best = best := by
  rw [innerTail4_3, scan, if_neg (by decide), innerCandidates4_3_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_4
    (if 3 ≤ innerCandidates4_3.length then search edge 3 [17, 383] innerCandidates4_3 400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan4_4

def innerTail4_2 : List Nat := 257 :: innerTail4_3
def innerCandidates4_2 : List Nat := [2027, 2243, 2741, 3023, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates4_2_checked : candidates edge 4 17 257 innerTail4_3 best.bound = innerCandidates4_2 := by
  unfold candidates
  have row : edge 257 = PrimePairCache.smallRow257.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_2 : search edge 3 [17, 257] innerCandidates4_2 274 best = best := by decide

theorem innerScan4_2 : scan edge (search edge 3) 4 [17] 17 innerTail4_2 best = best := by
  rw [innerTail4_2, scan, if_neg (by decide), innerCandidates4_2_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_3
    (if 3 ≤ innerCandidates4_2.length then search edge 3 [17, 257] innerCandidates4_2 274 best else best) = best
  rw [if_pos (by decide), innerChild4_2]
  exact innerScan4_3

def innerTail4_1 : List Nat := 239 :: innerTail4_2
def innerCandidates4_1 : List Nat := [1049, 4157, 4241]
set_option maxRecDepth 4096 in
theorem innerCandidates4_1_checked : candidates edge 4 17 239 innerTail4_2 best.bound = innerCandidates4_1 := by
  unfold candidates
  have row : edge 239 = PrimePairCache.smallRow239.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_1 : search edge 3 [17, 239] innerCandidates4_1 256 best = best := by decide

theorem innerScan4_1 : scan edge (search edge 3) 4 [17] 17 innerTail4_1 best = best := by
  rw [innerTail4_1, scan, if_neg (by decide), innerCandidates4_1_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_2
    (if 3 ≤ innerCandidates4_1.length then search edge 3 [17, 239] innerCandidates4_1 256 best else best) = best
  rw [if_pos (by decide), innerChild4_1]
  exact innerScan4_2

def innerTail4_0 : List Nat := 83 :: innerTail4_1
def innerCandidates4_0 : List Nat := [449, 1553, 2411, 3191, 3347, 5897, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates4_0_checked : candidates edge 4 17 83 innerTail4_1 best.bound = innerCandidates4_0 := by
  unfold candidates
  have row : edge 83 = PrimePairCache.smallRow83.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild4_0 : search edge 3 [17, 83] innerCandidates4_0 100 best = best := by decide

theorem innerScan4_0 : scan edge (search edge 3) 4 [17] 17 innerTail4_0 best = best := by
  rw [innerTail4_0, scan, if_neg (by decide), innerCandidates4_0_checked]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_1
    (if 3 ≤ innerCandidates4_0.length then search edge 3 [17, 83] innerCandidates4_0 100 best else best) = best
  rw [if_pos (by decide), innerChild4_0]
  exact innerScan4_1

theorem rootBranch4 : search edge 4 [17] rootCandidates4 17 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [17] 17 innerTail4_0 best = best
  exact innerScan4_0

end ProjectEulerSolutions.PrimePairUpper.Search
