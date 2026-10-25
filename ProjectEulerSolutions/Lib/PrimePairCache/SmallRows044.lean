import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2389_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2389_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2389 = smallRow2389_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2389_block_001 : List Nat := [2503, 2521, 2767, 3733]
set_option maxRecDepth 2048 in
theorem smallRow2389_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2389 = smallRow2389_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2389_block_002 : List Nat := [4243, 4957, 5023, 5167, 5479]
set_option maxRecDepth 2048 in
theorem smallRow2389_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2389 = smallRow2389_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2389_block_003 : List Nat := [6427, 6529, 6703, 6871, 6907, 7351, 7477, 7561]
set_option maxRecDepth 2048 in
theorem smallRow2389_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2389 = smallRow2389_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2389_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2389_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2389 = smallRow2389_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2389_small_left : List Nat := smallRow2389_block_000 ++ smallRow2389_block_001
theorem smallRow2389_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2389 = smallRow2389_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2389 ++ upperNeighbors PrimePairVertices.tree_block_001 2389 = _
  rw [smallRow2389_block_000_checked, smallRow2389_block_001_checked]
  rfl

def smallRow2389_small_right_right : List Nat := smallRow2389_block_003 ++ smallRow2389_block_004
theorem smallRow2389_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2389 = smallRow2389_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2389 ++ upperNeighbors PrimePairVertices.tree_block_004 2389 = _
  rw [smallRow2389_block_003_checked, smallRow2389_block_004_checked]
  rfl

def smallRow2389_small_right : List Nat := smallRow2389_block_002 ++ smallRow2389_small_right_right
theorem smallRow2389_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2389 = smallRow2389_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2389 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2389 = _
  rw [smallRow2389_block_002_checked, smallRow2389_small_right_right_checked]
  rfl

def smallRow2389_small : List Nat := smallRow2389_small_left ++ smallRow2389_small_right
theorem smallRow2389_small_checked : upperNeighbors PrimePairVertices.tree_small 2389 = smallRow2389_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2389 ++ upperNeighbors PrimePairVertices.tree_small_right 2389 = _
  rw [smallRow2389_small_left_checked, smallRow2389_small_right_checked]
  rfl

def smallRow2389 : List Nat := smallRow2389_small
theorem smallRow2389_checked : upperNeighbors PrimePairVertices.smallTree 2389 = smallRow2389 := smallRow2389_small_checked

def smallRow2393_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2393_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2393 = smallRow2393_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2393_block_001 : List Nat := [2411, 2423, 2663, 2927, 3083, 3347, 3491, 3557, 3821, 3851]
set_option maxRecDepth 2048 in
theorem smallRow2393_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2393 = smallRow2393_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2393_block_002 : List Nat := [4241, 4919, 5981]
set_option maxRecDepth 2048 in
theorem smallRow2393_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2393 = smallRow2393_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2393_block_003 : List Nat := [6329, 6791, 7127, 7829, 8069]
set_option maxRecDepth 2048 in
theorem smallRow2393_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2393 = smallRow2393_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2393_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2393_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2393 = smallRow2393_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2393_small_left : List Nat := smallRow2393_block_000 ++ smallRow2393_block_001
theorem smallRow2393_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2393 = smallRow2393_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2393 ++ upperNeighbors PrimePairVertices.tree_block_001 2393 = _
  rw [smallRow2393_block_000_checked, smallRow2393_block_001_checked]
  rfl

def smallRow2393_small_right_right : List Nat := smallRow2393_block_003 ++ smallRow2393_block_004
theorem smallRow2393_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2393 = smallRow2393_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2393 ++ upperNeighbors PrimePairVertices.tree_block_004 2393 = _
  rw [smallRow2393_block_003_checked, smallRow2393_block_004_checked]
  rfl

def smallRow2393_small_right : List Nat := smallRow2393_block_002 ++ smallRow2393_small_right_right
theorem smallRow2393_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2393 = smallRow2393_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2393 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2393 = _
  rw [smallRow2393_block_002_checked, smallRow2393_small_right_right_checked]
  rfl

def smallRow2393_small : List Nat := smallRow2393_small_left ++ smallRow2393_small_right
theorem smallRow2393_small_checked : upperNeighbors PrimePairVertices.tree_small 2393 = smallRow2393_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2393 ++ upperNeighbors PrimePairVertices.tree_small_right 2393 = _
  rw [smallRow2393_small_left_checked, smallRow2393_small_right_checked]
  rfl

def smallRow2393 : List Nat := smallRow2393_small
theorem smallRow2393_checked : upperNeighbors PrimePairVertices.smallTree 2393 = smallRow2393 := smallRow2393_small_checked

def smallRow2399_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2399_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2399 = smallRow2399_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2399_block_001 : List Nat := [2663, 3251, 3467, 3821, 3911]
set_option maxRecDepth 2048 in
theorem smallRow2399_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2399 = smallRow2399_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2399_block_002 : List Nat := [4457, 4517, 5717, 6131]
set_option maxRecDepth 2048 in
theorem smallRow2399_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2399 = smallRow2399_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2399_block_003 : List Nat := [6701, 6737, 6761, 6977, 7283, 8081, 8087]
set_option maxRecDepth 2048 in
theorem smallRow2399_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2399 = smallRow2399_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2399_block_004 : List Nat := [8363]
set_option maxRecDepth 2048 in
theorem smallRow2399_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2399 = smallRow2399_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2399_small_left : List Nat := smallRow2399_block_000 ++ smallRow2399_block_001
theorem smallRow2399_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2399 = smallRow2399_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2399 ++ upperNeighbors PrimePairVertices.tree_block_001 2399 = _
  rw [smallRow2399_block_000_checked, smallRow2399_block_001_checked]
  rfl

def smallRow2399_small_right_right : List Nat := smallRow2399_block_003 ++ smallRow2399_block_004
theorem smallRow2399_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2399 = smallRow2399_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2399 ++ upperNeighbors PrimePairVertices.tree_block_004 2399 = _
  rw [smallRow2399_block_003_checked, smallRow2399_block_004_checked]
  rfl

def smallRow2399_small_right : List Nat := smallRow2399_block_002 ++ smallRow2399_small_right_right
theorem smallRow2399_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2399 = smallRow2399_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2399 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2399 = _
  rw [smallRow2399_block_002_checked, smallRow2399_small_right_right_checked]
  rfl

def smallRow2399_small : List Nat := smallRow2399_small_left ++ smallRow2399_small_right
theorem smallRow2399_small_checked : upperNeighbors PrimePairVertices.tree_small 2399 = smallRow2399_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2399 ++ upperNeighbors PrimePairVertices.tree_small_right 2399 = _
  rw [smallRow2399_small_left_checked, smallRow2399_small_right_checked]
  rfl

def smallRow2399 : List Nat := smallRow2399_small
theorem smallRow2399_checked : upperNeighbors PrimePairVertices.smallTree 2399 = smallRow2399 := smallRow2399_small_checked

def smallRow2411_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2411_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2411 = smallRow2411_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2411_block_001 : List Nat := [3917]
set_option maxRecDepth 2048 in
theorem smallRow2411_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2411 = smallRow2411_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2411_block_002 : List Nat := [4241, 4253, 4793, 4919, 5171, 5519, 5879]
set_option maxRecDepth 2048 in
theorem smallRow2411_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2411 = smallRow2411_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2411_block_003 : List Nat := [7019, 7433, 7481]
set_option maxRecDepth 2048 in
theorem smallRow2411_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2411 = smallRow2411_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2411_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow2411_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2411 = smallRow2411_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2411_small_left : List Nat := smallRow2411_block_000 ++ smallRow2411_block_001
theorem smallRow2411_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2411 = smallRow2411_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2411 ++ upperNeighbors PrimePairVertices.tree_block_001 2411 = _
  rw [smallRow2411_block_000_checked, smallRow2411_block_001_checked]
  rfl

def smallRow2411_small_right_right : List Nat := smallRow2411_block_003 ++ smallRow2411_block_004
theorem smallRow2411_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2411 = smallRow2411_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2411 ++ upperNeighbors PrimePairVertices.tree_block_004 2411 = _
  rw [smallRow2411_block_003_checked, smallRow2411_block_004_checked]
  rfl

def smallRow2411_small_right : List Nat := smallRow2411_block_002 ++ smallRow2411_small_right_right
theorem smallRow2411_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2411 = smallRow2411_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2411 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2411 = _
  rw [smallRow2411_block_002_checked, smallRow2411_small_right_right_checked]
  rfl

def smallRow2411_small : List Nat := smallRow2411_small_left ++ smallRow2411_small_right
theorem smallRow2411_small_checked : upperNeighbors PrimePairVertices.tree_small 2411 = smallRow2411_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2411 ++ upperNeighbors PrimePairVertices.tree_small_right 2411 = _
  rw [smallRow2411_small_left_checked, smallRow2411_small_right_checked]
  rfl

def smallRow2411 : List Nat := smallRow2411_small
theorem smallRow2411_checked : upperNeighbors PrimePairVertices.smallTree 2411 = smallRow2411 := smallRow2411_small_checked

def smallRow2417_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2417_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2417 = smallRow2417_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2417_block_001 : List Nat := [2741, 3449, 4001]
set_option maxRecDepth 2048 in
theorem smallRow2417_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2417 = smallRow2417_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2417_block_002 : List Nat := [4229, 4253, 5153, 5303, 5903, 5927]
set_option maxRecDepth 2048 in
theorem smallRow2417_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2417 = smallRow2417_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2417_block_003 : List Nat := [6389, 6563, 6599, 6857, 7019, 7103, 7691, 7877]
set_option maxRecDepth 2048 in
theorem smallRow2417_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2417 = smallRow2417_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2417_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow2417_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2417 = smallRow2417_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2417_small_left : List Nat := smallRow2417_block_000 ++ smallRow2417_block_001
theorem smallRow2417_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2417 = smallRow2417_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2417 ++ upperNeighbors PrimePairVertices.tree_block_001 2417 = _
  rw [smallRow2417_block_000_checked, smallRow2417_block_001_checked]
  rfl

def smallRow2417_small_right_right : List Nat := smallRow2417_block_003 ++ smallRow2417_block_004
theorem smallRow2417_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2417 = smallRow2417_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2417 ++ upperNeighbors PrimePairVertices.tree_block_004 2417 = _
  rw [smallRow2417_block_003_checked, smallRow2417_block_004_checked]
  rfl

def smallRow2417_small_right : List Nat := smallRow2417_block_002 ++ smallRow2417_small_right_right
theorem smallRow2417_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2417 = smallRow2417_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2417 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2417 = _
  rw [smallRow2417_block_002_checked, smallRow2417_small_right_right_checked]
  rfl

def smallRow2417_small : List Nat := smallRow2417_small_left ++ smallRow2417_small_right
theorem smallRow2417_small_checked : upperNeighbors PrimePairVertices.tree_small 2417 = smallRow2417_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2417 ++ upperNeighbors PrimePairVertices.tree_small_right 2417 = _
  rw [smallRow2417_small_left_checked, smallRow2417_small_right_checked]
  rfl

def smallRow2417 : List Nat := smallRow2417_small
theorem smallRow2417_checked : upperNeighbors PrimePairVertices.smallTree 2417 = smallRow2417 := smallRow2417_small_checked

def smallRow2423_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2423_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2423 = smallRow2423_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2423_block_001 : List Nat := [3359, 3851, 3863]
set_option maxRecDepth 2048 in
theorem smallRow2423_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2423 = smallRow2423_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2423_block_002 : List Nat := [4391, 4451, 4547, 5021, 5147, 5303, 5639, 5669]
set_option maxRecDepth 2048 in
theorem smallRow2423_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2423 = smallRow2423_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2423_block_003 : List Nat := [7727, 7919]
set_option maxRecDepth 2048 in
theorem smallRow2423_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2423 = smallRow2423_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2423_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow2423_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2423 = smallRow2423_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2423_small_left : List Nat := smallRow2423_block_000 ++ smallRow2423_block_001
theorem smallRow2423_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2423 = smallRow2423_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2423 ++ upperNeighbors PrimePairVertices.tree_block_001 2423 = _
  rw [smallRow2423_block_000_checked, smallRow2423_block_001_checked]
  rfl

def smallRow2423_small_right_right : List Nat := smallRow2423_block_003 ++ smallRow2423_block_004
theorem smallRow2423_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2423 = smallRow2423_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2423 ++ upperNeighbors PrimePairVertices.tree_block_004 2423 = _
  rw [smallRow2423_block_003_checked, smallRow2423_block_004_checked]
  rfl

def smallRow2423_small_right : List Nat := smallRow2423_block_002 ++ smallRow2423_small_right_right
theorem smallRow2423_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2423 = smallRow2423_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2423 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2423 = _
  rw [smallRow2423_block_002_checked, smallRow2423_small_right_right_checked]
  rfl

def smallRow2423_small : List Nat := smallRow2423_small_left ++ smallRow2423_small_right
theorem smallRow2423_small_checked : upperNeighbors PrimePairVertices.tree_small 2423 = smallRow2423_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2423 ++ upperNeighbors PrimePairVertices.tree_small_right 2423 = _
  rw [smallRow2423_small_left_checked, smallRow2423_small_right_checked]
  rfl

def smallRow2423 : List Nat := smallRow2423_small
theorem smallRow2423_checked : upperNeighbors PrimePairVertices.smallTree 2423 = smallRow2423 := smallRow2423_small_checked

def smallRow2437_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2437_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2437 = smallRow2437_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2437_block_001 : List Nat := [3301, 3361, 3541, 3889, 3931]
set_option maxRecDepth 2048 in
theorem smallRow2437_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2437 = smallRow2437_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2437_block_002 : List Nat := [4111, 4621, 4969, 5077, 6073]
set_option maxRecDepth 2048 in
theorem smallRow2437_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2437 = smallRow2437_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2437_block_003 : List Nat := [6301, 6337, 6607, 6967, 7027, 7039]
set_option maxRecDepth 2048 in
theorem smallRow2437_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2437 = smallRow2437_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2437_block_004 : List Nat := [8353]
set_option maxRecDepth 2048 in
theorem smallRow2437_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2437 = smallRow2437_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2437_small_left : List Nat := smallRow2437_block_000 ++ smallRow2437_block_001
theorem smallRow2437_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2437 = smallRow2437_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2437 ++ upperNeighbors PrimePairVertices.tree_block_001 2437 = _
  rw [smallRow2437_block_000_checked, smallRow2437_block_001_checked]
  rfl

def smallRow2437_small_right_right : List Nat := smallRow2437_block_003 ++ smallRow2437_block_004
theorem smallRow2437_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2437 = smallRow2437_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2437 ++ upperNeighbors PrimePairVertices.tree_block_004 2437 = _
  rw [smallRow2437_block_003_checked, smallRow2437_block_004_checked]
  rfl

def smallRow2437_small_right : List Nat := smallRow2437_block_002 ++ smallRow2437_small_right_right
theorem smallRow2437_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2437 = smallRow2437_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2437 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2437 = _
  rw [smallRow2437_block_002_checked, smallRow2437_small_right_right_checked]
  rfl

def smallRow2437_small : List Nat := smallRow2437_small_left ++ smallRow2437_small_right
theorem smallRow2437_small_checked : upperNeighbors PrimePairVertices.tree_small 2437 = smallRow2437_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2437 ++ upperNeighbors PrimePairVertices.tree_small_right 2437 = _
  rw [smallRow2437_small_left_checked, smallRow2437_small_right_checked]
  rfl

def smallRow2437 : List Nat := smallRow2437_small
theorem smallRow2437_checked : upperNeighbors PrimePairVertices.smallTree 2437 = smallRow2437 := smallRow2437_small_checked

def smallRow2441_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2441_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2441 = smallRow2441_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2441_block_001 : List Nat := [2543, 3209, 3671, 3677]
set_option maxRecDepth 2048 in
theorem smallRow2441_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2441 = smallRow2441_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2441_block_002 : List Nat := [4157, 4349, 4787, 5087, 5147, 5297, 5333, 5381, 6089, 6113]
set_option maxRecDepth 2048 in
theorem smallRow2441_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2441 = smallRow2441_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2441_block_003 : List Nat := [6659, 7121, 7643, 7691, 7727]
set_option maxRecDepth 2048 in
theorem smallRow2441_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2441 = smallRow2441_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2441_block_004 : List Nat := [8231]
set_option maxRecDepth 2048 in
theorem smallRow2441_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2441 = smallRow2441_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2441_small_left : List Nat := smallRow2441_block_000 ++ smallRow2441_block_001
theorem smallRow2441_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2441 = smallRow2441_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2441 ++ upperNeighbors PrimePairVertices.tree_block_001 2441 = _
  rw [smallRow2441_block_000_checked, smallRow2441_block_001_checked]
  rfl

def smallRow2441_small_right_right : List Nat := smallRow2441_block_003 ++ smallRow2441_block_004
theorem smallRow2441_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2441 = smallRow2441_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2441 ++ upperNeighbors PrimePairVertices.tree_block_004 2441 = _
  rw [smallRow2441_block_003_checked, smallRow2441_block_004_checked]
  rfl

def smallRow2441_small_right : List Nat := smallRow2441_block_002 ++ smallRow2441_small_right_right
theorem smallRow2441_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2441 = smallRow2441_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2441 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2441 = _
  rw [smallRow2441_block_002_checked, smallRow2441_small_right_right_checked]
  rfl

def smallRow2441_small : List Nat := smallRow2441_small_left ++ smallRow2441_small_right
theorem smallRow2441_small_checked : upperNeighbors PrimePairVertices.tree_small 2441 = smallRow2441_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2441 ++ upperNeighbors PrimePairVertices.tree_small_right 2441 = _
  rw [smallRow2441_small_left_checked, smallRow2441_small_right_checked]
  rfl

def smallRow2441 : List Nat := smallRow2441_small
theorem smallRow2441_checked : upperNeighbors PrimePairVertices.smallTree 2441 = smallRow2441 := smallRow2441_small_checked

end ProjectEulerSolutions.PrimePairCache
