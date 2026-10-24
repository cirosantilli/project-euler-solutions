import ProjectEulerSolutions.Lib.PrimePairSearch.SmallP3Chunk001
import ProjectEulerSolutions.Lib.CliqueSearch

/-! Generated scan steps for the first small-domain branch. -/
namespace ProjectEulerSolutions.PrimePairSearch
open PrimePairCertificate CliqueSearch

def smallTail60 : List Nat := 2843 :: smallTail61
def smallCandidates60 : List Nat := [4253, 6899]
set_option maxRecDepth 2048 in
theorem smallCandidates60_checked : candidates relation 4 3 2843 smallTail61 smallBest.bound = smallCandidates60 := by decide

theorem smallScan60 : scan relation (search relation 3) 4 [3] 3 smallTail60 smallBest = smallBest := by
  rw [smallTail60, scan, if_neg (by decide), smallCandidates60_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail61
    (if 3 ≤ smallCandidates60.length then search relation 3 [3, 2843] smallCandidates60 2846 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan61

def smallTail59 : List Nat := 2789 :: smallTail60
def smallCandidates59 : List Nat := [3461, 5051, 6563, 7691]
set_option maxRecDepth 2048 in
theorem smallCandidates59_checked : candidates relation 4 3 2789 smallTail60 smallBest.bound = smallCandidates59 := by decide

set_option maxRecDepth 2048 in
theorem smallChild59 : search relation 3 [3, 2789] smallCandidates59 2792 smallBest = smallBest := by decide

theorem smallScan59 : scan relation (search relation 3) 4 [3] 3 smallTail59 smallBest = smallBest := by
  rw [smallTail59, scan, if_neg (by decide), smallCandidates59_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail60
    (if 3 ≤ smallCandidates59.length then search relation 3 [3, 2789] smallCandidates59 2792 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild59]
  exact smallScan60

def smallTail58 : List Nat := 2707 :: smallTail59
def smallCandidates58 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates58_checked : candidates relation 4 3 2707 smallTail59 smallBest.bound = smallCandidates58 := by decide

theorem smallScan58 : scan relation (search relation 3) 4 [3] 3 smallTail58 smallBest = smallBest := by
  rw [smallTail58, scan, if_neg (by decide), smallCandidates58_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail59
    (if 3 ≤ smallCandidates58.length then search relation 3 [3, 2707] smallCandidates58 2710 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan59

def smallTail57 : List Nat := 2579 :: smallTail58
def smallCandidates57 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates57_checked : candidates relation 4 3 2579 smallTail58 smallBest.bound = smallCandidates57 := by decide

theorem smallScan57 : scan relation (search relation 3) 4 [3] 3 smallTail57 smallBest = smallBest := by
  rw [smallTail57, scan, if_neg (by decide), smallCandidates57_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail58
    (if 3 ≤ smallCandidates57.length then search relation 3 [3, 2579] smallCandidates57 2582 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan58

def smallTail56 : List Nat := 2503 :: smallTail57
def smallCandidates56 : List Nat := [5281, 5869, 7591]
set_option maxRecDepth 2048 in
theorem smallCandidates56_checked : candidates relation 4 3 2503 smallTail57 smallBest.bound = smallCandidates56 := by decide

set_option maxRecDepth 2048 in
theorem smallChild56 : search relation 3 [3, 2503] smallCandidates56 2506 smallBest = smallBest := by decide

theorem smallScan56 : scan relation (search relation 3) 4 [3] 3 smallTail56 smallBest = smallBest := by
  rw [smallTail56, scan, if_neg (by decide), smallCandidates56_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail57
    (if 3 ≤ smallCandidates56.length then search relation 3 [3, 2503] smallCandidates56 2506 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild56]
  exact smallScan57

def smallTail55 : List Nat := 2441 :: smallTail56
def smallCandidates55 : List Nat := [5381, 7691]
set_option maxRecDepth 2048 in
theorem smallCandidates55_checked : candidates relation 4 3 2441 smallTail56 smallBest.bound = smallCandidates55 := by decide

theorem smallScan55 : scan relation (search relation 3) 4 [3] 3 smallTail55 smallBest = smallBest := by
  rw [smallTail55, scan, if_neg (by decide), smallCandidates55_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail56
    (if 3 ≤ smallCandidates55.length then search relation 3 [3, 2441] smallCandidates55 2444 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan56

def smallTail54 : List Nat := 2411 :: smallTail55
def smallCandidates54 : List Nat := [4253, 4919, 5171]
set_option maxRecDepth 2048 in
theorem smallCandidates54_checked : candidates relation 4 3 2411 smallTail55 smallBest.bound = smallCandidates54 := by decide

set_option maxRecDepth 2048 in
theorem smallChild54 : search relation 3 [3, 2411] smallCandidates54 2414 smallBest = smallBest := by decide

theorem smallScan54 : scan relation (search relation 3) 4 [3] 3 smallTail54 smallBest = smallBest := by
  rw [smallTail54, scan, if_neg (by decide), smallCandidates54_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail55
    (if 3 ≤ smallCandidates54.length then search relation 3 [3, 2411] smallCandidates54 2414 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild54]
  exact smallScan55

def smallTail53 : List Nat := 2381 :: smallTail54
def smallCandidates53 : List Nat := [6563]
set_option maxRecDepth 2048 in
theorem smallCandidates53_checked : candidates relation 4 3 2381 smallTail54 smallBest.bound = smallCandidates53 := by decide

theorem smallScan53 : scan relation (search relation 3) 4 [3] 3 smallTail53 smallBest = smallBest := by
  rw [smallTail53, scan, if_neg (by decide), smallCandidates53_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail54
    (if 3 ≤ smallCandidates53.length then search relation 3 [3, 2381] smallCandidates53 2384 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan54

def smallTail52 : List Nat := 2377 :: smallTail53
def smallCandidates52 : List Nat := [4159, 6073]
set_option maxRecDepth 2048 in
theorem smallCandidates52_checked : candidates relation 4 3 2377 smallTail53 smallBest.bound = smallCandidates52 := by decide

theorem smallScan52 : scan relation (search relation 3) 4 [3] 3 smallTail52 smallBest = smallBest := by
  rw [smallTail52, scan, if_neg (by decide), smallCandidates52_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail53
    (if 3 ≤ smallCandidates52.length then search relation 3 [3, 2377] smallCandidates52 2380 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan53

def smallTail51 : List Nat := 2297 :: smallTail52
def smallCandidates51 : List Nat := [2381]
set_option maxRecDepth 2048 in
theorem smallCandidates51_checked : candidates relation 4 3 2297 smallTail52 smallBest.bound = smallCandidates51 := by decide

theorem smallScan51 : scan relation (search relation 3) 4 [3] 3 smallTail51 smallBest = smallBest := by
  rw [smallTail51, scan, if_neg (by decide), smallCandidates51_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail52
    (if 3 ≤ smallCandidates51.length then search relation 3 [3, 2297] smallCandidates51 2300 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan52

def smallTail50 : List Nat := 2213 :: smallTail51
def smallCandidates50 : List Nat := [4253]
set_option maxRecDepth 2048 in
theorem smallCandidates50_checked : candidates relation 4 3 2213 smallTail51 smallBest.bound = smallCandidates50 := by decide

theorem smallScan50 : scan relation (search relation 3) 4 [3] 3 smallTail50 smallBest = smallBest := by
  rw [smallTail50, scan, if_neg (by decide), smallCandidates50_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail51
    (if 3 ≤ smallCandidates50.length then search relation 3 [3, 2213] smallCandidates50 2216 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan51

def smallTail49 : List Nat := 2143 :: smallTail50
def smallCandidates49 : List Nat := [6871]
set_option maxRecDepth 2048 in
theorem smallCandidates49_checked : candidates relation 4 3 2143 smallTail50 smallBest.bound = smallCandidates49 := by decide

theorem smallScan49 : scan relation (search relation 3) 4 [3] 3 smallTail49 smallBest = smallBest := by
  rw [smallTail49, scan, if_neg (by decide), smallCandidates49_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail50
    (if 3 ≤ smallCandidates49.length then search relation 3 [3, 2143] smallCandidates49 2146 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan50

def smallTail48 : List Nat := 2069 :: smallTail49
def smallCandidates48 : List Nat := [2297, 7253, 8219]
set_option maxRecDepth 2048 in
theorem smallCandidates48_checked : candidates relation 4 3 2069 smallTail49 smallBest.bound = smallCandidates48 := by decide

set_option maxRecDepth 2048 in
theorem smallChild48 : search relation 3 [3, 2069] smallCandidates48 2072 smallBest = smallBest := by decide

theorem smallScan48 : scan relation (search relation 3) 4 [3] 3 smallTail48 smallBest = smallBest := by
  rw [smallTail48, scan, if_neg (by decide), smallCandidates48_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail49
    (if 3 ≤ smallCandidates48.length then search relation 3 [3, 2069] smallCandidates48 2072 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild48]
  exact smallScan49

def smallTail47 : List Nat := 1907 :: smallTail48
def smallCandidates47 : List Nat := [5099]
set_option maxRecDepth 2048 in
theorem smallCandidates47_checked : candidates relation 4 3 1907 smallTail48 smallBest.bound = smallCandidates47 := by decide

theorem smallScan47 : scan relation (search relation 3) 4 [3] 3 smallTail47 smallBest = smallBest := by
  rw [smallTail47, scan, if_neg (by decide), smallCandidates47_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail48
    (if 3 ≤ smallCandidates47.length then search relation 3 [3, 1907] smallCandidates47 1910 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan48

def smallTail46 : List Nat := 1699 :: smallTail47
def smallCandidates46 : List Nat := [6571]
set_option maxRecDepth 2048 in
theorem smallCandidates46_checked : candidates relation 4 3 1699 smallTail47 smallBest.bound = smallCandidates46 := by decide

theorem smallScan46 : scan relation (search relation 3) 4 [3] 3 smallTail46 smallBest = smallBest := by
  rw [smallTail46, scan, if_neg (by decide), smallCandidates46_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail47
    (if 3 ≤ smallCandidates46.length then search relation 3 [3, 1699] smallCandidates46 1702 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan47

def smallTail45 : List Nat := 1667 :: smallTail46
def smallCandidates45 : List Nat := [5051, 5531, 5573, 6353]
set_option maxRecDepth 2048 in
theorem smallCandidates45_checked : candidates relation 4 3 1667 smallTail46 smallBest.bound = smallCandidates45 := by decide

set_option maxRecDepth 2048 in
theorem smallChild45 : search relation 3 [3, 1667] smallCandidates45 1670 smallBest = smallBest := by decide

theorem smallScan45 : scan relation (search relation 3) 4 [3] 3 smallTail45 smallBest = smallBest := by
  rw [smallTail45, scan, if_neg (by decide), smallCandidates45_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail46
    (if 3 ≤ smallCandidates45.length then search relation 3 [3, 1667] smallCandidates45 1670 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild45]
  exact smallScan46

def smallTail44 : List Nat := 1663 :: smallTail45
def smallCandidates44 : List Nat := [5449, 7963]
set_option maxRecDepth 2048 in
theorem smallCandidates44_checked : candidates relation 4 3 1663 smallTail45 smallBest.bound = smallCandidates44 := by decide

theorem smallScan44 : scan relation (search relation 3) 4 [3] 3 smallTail44 smallBest = smallBest := by
  rw [smallTail44, scan, if_neg (by decide), smallCandidates44_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail45
    (if 3 ≤ smallCandidates44.length then search relation 3 [3, 1663] smallCandidates44 1666 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan45

def smallTail43 : List Nat := 1657 :: smallTail44
def smallCandidates43 : List Nat := [1699, 2917, 5059]
set_option maxRecDepth 2048 in
theorem smallCandidates43_checked : candidates relation 4 3 1657 smallTail44 smallBest.bound = smallCandidates43 := by decide

set_option maxRecDepth 2048 in
theorem smallChild43 : search relation 3 [3, 1657] smallCandidates43 1660 smallBest = smallBest := by decide

theorem smallScan43 : scan relation (search relation 3) 4 [3] 3 smallTail43 smallBest = smallBest := by
  rw [smallTail43, scan, if_neg (by decide), smallCandidates43_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail44
    (if 3 ≤ smallCandidates43.length then search relation 3 [3, 1657] smallCandidates43 1660 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild43]
  exact smallScan44

def smallTail42 : List Nat := 1627 :: smallTail43
def smallCandidates42 : List Nat := [2917, 3637, 3769, 6277, 7159]
set_option maxRecDepth 2048 in
theorem smallCandidates42_checked : candidates relation 4 3 1627 smallTail43 smallBest.bound = smallCandidates42 := by decide

set_option maxRecDepth 2048 in
theorem smallChild42 : search relation 3 [3, 1627] smallCandidates42 1630 smallBest = smallBest := by decide

theorem smallScan42 : scan relation (search relation 3) 4 [3] 3 smallTail42 smallBest = smallBest := by
  rw [smallTail42, scan, if_neg (by decide), smallCandidates42_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail43
    (if 3 ≤ smallCandidates42.length then search relation 3 [3, 1627] smallCandidates42 1630 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild42]
  exact smallScan43

def smallTail41 : List Nat := 1607 :: smallTail42
def smallCandidates41 : List Nat := [5099]
set_option maxRecDepth 2048 in
theorem smallCandidates41_checked : candidates relation 4 3 1607 smallTail42 smallBest.bound = smallCandidates41 := by decide

theorem smallScan41 : scan relation (search relation 3) 4 [3] 3 smallTail41 smallBest = smallBest := by
  rw [smallTail41, scan, if_neg (by decide), smallCandidates41_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail42
    (if 3 ≤ smallCandidates41.length then search relation 3 [3, 1607] smallCandidates41 1610 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan42

def smallTail40 : List Nat := 1531 :: smallTail41
def smallCandidates40 : List Nat := [5059, 5281]
set_option maxRecDepth 2048 in
theorem smallCandidates40_checked : candidates relation 4 3 1531 smallTail41 smallBest.bound = smallCandidates40 := by decide

theorem smallScan40 : scan relation (search relation 3) 4 [3] 3 smallTail40 smallBest = smallBest := by
  rw [smallTail40, scan, if_neg (by decide), smallCandidates40_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail41
    (if 3 ≤ smallCandidates40.length then search relation 3 [3, 1531] smallCandidates40 1534 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan41

def smallTail39 : List Nat := 1481 :: smallTail40
def smallCandidates39 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates39_checked : candidates relation 4 3 1481 smallTail40 smallBest.bound = smallCandidates39 := by decide

theorem smallScan39 : scan relation (search relation 3) 4 [3] 3 smallTail39 smallBest = smallBest := by
  rw [smallTail39, scan, if_neg (by decide), smallCandidates39_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail40
    (if 3 ≤ smallCandidates39.length then search relation 3 [3, 1481] smallCandidates39 1484 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan40

def smallTail38 : List Nat := 1237 :: smallTail39
def smallCandidates38 : List Nat := [1657, 6571]
set_option maxRecDepth 2048 in
theorem smallCandidates38_checked : candidates relation 4 3 1237 smallTail39 smallBest.bound = smallCandidates38 := by decide

theorem smallScan38 : scan relation (search relation 3) 4 [3] 3 smallTail38 smallBest = smallBest := by
  rw [smallTail38, scan, if_neg (by decide), smallCandidates38_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail39
    (if 3 ≤ smallCandidates38.length then search relation 3 [3, 1237] smallCandidates38 1240 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan39

def smallTail37 : List Nat := 1193 :: smallTail38
def smallCandidates37 : List Nat := [2441, 3923, 5507, 5801]
set_option maxRecDepth 2048 in
theorem smallCandidates37_checked : candidates relation 4 3 1193 smallTail38 smallBest.bound = smallCandidates37 := by decide

set_option maxRecDepth 2048 in
theorem smallChild37 : search relation 3 [3, 1193] smallCandidates37 1196 smallBest = smallBest := by decide

theorem smallScan37 : scan relation (search relation 3) 4 [3] 3 smallTail37 smallBest = smallBest := by
  rw [smallTail37, scan, if_neg (by decide), smallCandidates37_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail38
    (if 3 ≤ smallCandidates37.length then search relation 3 [3, 1193] smallCandidates37 1196 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild37]
  exact smallScan38

def smallTail36 : List Nat := 1181 :: smallTail37
def smallCandidates36 : List Nat := [5171, 8237]
set_option maxRecDepth 2048 in
theorem smallCandidates36_checked : candidates relation 4 3 1181 smallTail37 smallBest.bound = smallCandidates36 := by decide

theorem smallScan36 : scan relation (search relation 3) 4 [3] 3 smallTail36 smallBest = smallBest := by
  rw [smallTail36, scan, if_neg (by decide), smallCandidates36_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail37
    (if 3 ≤ smallCandidates36.length then search relation 3 [3, 1181] smallCandidates36 1184 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan37

def smallTail35 : List Nat := 1051 :: smallTail36
def smallCandidates35 : List Nat := [4729, 5839]
set_option maxRecDepth 2048 in
theorem smallCandidates35_checked : candidates relation 4 3 1051 smallTail36 smallBest.bound = smallCandidates35 := by decide

theorem smallScan35 : scan relation (search relation 3) 4 [3] 3 smallTail35 smallBest = smallBest := by
  rw [smallTail35, scan, if_neg (by decide), smallCandidates35_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail36
    (if 3 ≤ smallCandidates35.length then search relation 3 [3, 1051] smallCandidates35 1054 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan36

def smallTail34 : List Nat := 1033 :: smallTail35
def smallCandidates34 : List Nat := [1663, 5869, 6451]
set_option maxRecDepth 2048 in
theorem smallCandidates34_checked : candidates relation 4 3 1033 smallTail35 smallBest.bound = smallCandidates34 := by decide

set_option maxRecDepth 2048 in
theorem smallChild34 : search relation 3 [3, 1033] smallCandidates34 1036 smallBest = smallBest := by decide

theorem smallScan34 : scan relation (search relation 3) 4 [3] 3 smallTail34 smallBest = smallBest := by
  rw [smallTail34, scan, if_neg (by decide), smallCandidates34_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail35
    (if 3 ≤ smallCandidates34.length then search relation 3 [3, 1033] smallCandidates34 1036 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild34]
  exact smallScan35

def smallTail33 : List Nat := 1019 :: smallTail34
def smallCandidates33 : List Nat := [2957, 6947]
set_option maxRecDepth 2048 in
theorem smallCandidates33_checked : candidates relation 4 3 1019 smallTail34 smallBest.bound = smallCandidates33 := by decide

theorem smallScan33 : scan relation (search relation 3) 4 [3] 3 smallTail33 smallBest = smallBest := by
  rw [smallTail33, scan, if_neg (by decide), smallCandidates33_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail34
    (if 3 ≤ smallCandidates33.length then search relation 3 [3, 1019] smallCandidates33 1022 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan34

def smallTail32 : List Nat := 1013 :: smallTail33
def smallCandidates32 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates32_checked : candidates relation 4 3 1013 smallTail33 smallBest.bound = smallCandidates32 := by decide

theorem smallScan32 : scan relation (search relation 3) 4 [3] 3 smallTail32 smallBest = smallBest := by
  rw [smallTail32, scan, if_neg (by decide), smallCandidates32_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail33
    (if 3 ≤ smallCandidates32.length then search relation 3 [3, 1013] smallCandidates32 1016 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan33

def smallTail31 : List Nat := 947 :: smallTail32
def smallCandidates31 : List Nat := [5051, 6947]
set_option maxRecDepth 2048 in
theorem smallCandidates31_checked : candidates relation 4 3 947 smallTail32 smallBest.bound = smallCandidates31 := by decide

theorem smallScan31 : scan relation (search relation 3) 4 [3] 3 smallTail31 smallBest = smallBest := by
  rw [smallTail31, scan, if_neg (by decide), smallCandidates31_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail32
    (if 3 ≤ smallCandidates31.length then search relation 3 [3, 947] smallCandidates31 950 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan32

def smallTail30 : List Nat := 929 :: smallTail31
def smallCandidates30 : List Nat := [6563]
set_option maxRecDepth 2048 in
theorem smallCandidates30_checked : candidates relation 4 3 929 smallTail31 smallBest.bound = smallCandidates30 := by decide

theorem smallScan30 : scan relation (search relation 3) 4 [3] 3 smallTail30 smallBest = smallBest := by
  rw [smallTail30, scan, if_neg (by decide), smallCandidates30_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail31
    (if 3 ≤ smallCandidates30.length then search relation 3 [3, 929] smallCandidates30 932 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan31

def smallTail29 : List Nat := 823 :: smallTail30
def smallCandidates29 : List Nat := [5281, 5323]
set_option maxRecDepth 2048 in
theorem smallCandidates29_checked : candidates relation 4 3 823 smallTail30 smallBest.bound = smallCandidates29 := by decide

theorem smallScan29 : scan relation (search relation 3) 4 [3] 3 smallTail29 smallBest = smallBest := by
  rw [smallTail29, scan, if_neg (by decide), smallCandidates29_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail30
    (if 3 ≤ smallCandidates29.length then search relation 3 [3, 823] smallCandidates29 826 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan30

end ProjectEulerSolutions.PrimePairSearch
