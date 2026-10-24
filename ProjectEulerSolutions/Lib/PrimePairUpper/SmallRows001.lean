import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairUpperBound

/-! Generated possible-edge rows; every excluded edge is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairUpper
open PrimePairCertificate

def smallRow2711_block_000 : List Nat := []
def smallPlan2711_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2711_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2711 smallRow2711_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2711_block_000
  · decide
  · rfl

def smallRow2711_block_001 : List Nat := [2963, 3137, 3413, 3851]
def smallPlan2711_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow2711_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2711 smallRow2711_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2711_block_001
  · decide
  · rfl

def smallRow2711_block_002 : List Nat := [4211, 4421, 5441, 5651, 5741, 6113]
def smallPlan2711_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2711_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2711 smallRow2711_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2711_block_002
  · decide
  · rfl

def smallRow2711_block_003 : List Nat := [6491, 7151, 7457, 7589, 7817, 7877]
def smallPlan2711_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2711_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2711 smallRow2711_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2711_block_003
  · decide
  · rfl

def smallRow2711_block_004 : List Nat := []
def smallPlan2711_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2711_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2711 smallRow2711_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2711_block_004
  · decide
  · rfl

def smallRow2711_small_left : List Nat := smallRow2711_block_000 ++ smallRow2711_block_001
theorem smallRow2711_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2711 smallRow2711_small_left :=
  RowUpperBound.branch smallRow2711_block_000_covers smallRow2711_block_001_covers

def smallRow2711_small_right_right : List Nat := smallRow2711_block_003 ++ smallRow2711_block_004
theorem smallRow2711_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2711 smallRow2711_small_right_right :=
  RowUpperBound.branch smallRow2711_block_003_covers smallRow2711_block_004_covers

def smallRow2711_small_right : List Nat := smallRow2711_block_002 ++ smallRow2711_small_right_right
theorem smallRow2711_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2711 smallRow2711_small_right :=
  RowUpperBound.branch smallRow2711_block_002_covers smallRow2711_small_right_right_covers

def smallRow2711_small : List Nat := smallRow2711_small_left ++ smallRow2711_small_right
theorem smallRow2711_small_covers : RowUpperBound PrimePairVertices.tree_small 2711 smallRow2711_small :=
  RowUpperBound.branch smallRow2711_small_left_covers smallRow2711_small_right_covers

def smallRow2711 : List Nat := smallRow2711_small
theorem smallRow2711_covers : RowUpperBound PrimePairVertices.smallTree 2711 smallRow2711 := smallRow2711_small_covers

def smallRow2713_block_000 : List Nat := []
def smallPlan2713_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2713_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2713 smallRow2713_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2713_block_000
  · decide
  · rfl

def smallRow2713_block_001 : List Nat := [2797, 3433, 3517, 3739, 3889]
def smallPlan2713_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .second, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2713_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2713 smallRow2713_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2713_block_001
  · decide
  · rfl

def smallRow2713_block_002 : List Nat := [5443, 5749, 5869]
def smallPlan2713_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2713_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2713 smallRow2713_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2713_block_002
  · decide
  · rfl

def smallRow2713_block_003 : List Nat := [6247, 6469, 7207, 7213, 8191]
def smallPlan2713_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible]
set_option maxRecDepth 2048 in
theorem smallRow2713_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2713 smallRow2713_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2713_block_003
  · decide
  · rfl

def smallRow2713_block_004 : List Nat := []
def smallPlan2713_block_004 : List RowChoice := [.first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2713_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2713 smallRow2713_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2713_block_004
  · decide
  · rfl

def smallRow2713_small_left : List Nat := smallRow2713_block_000 ++ smallRow2713_block_001
theorem smallRow2713_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2713 smallRow2713_small_left :=
  RowUpperBound.branch smallRow2713_block_000_covers smallRow2713_block_001_covers

def smallRow2713_small_right_right : List Nat := smallRow2713_block_003 ++ smallRow2713_block_004
theorem smallRow2713_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2713 smallRow2713_small_right_right :=
  RowUpperBound.branch smallRow2713_block_003_covers smallRow2713_block_004_covers

def smallRow2713_small_right : List Nat := smallRow2713_block_002 ++ smallRow2713_small_right_right
theorem smallRow2713_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2713 smallRow2713_small_right :=
  RowUpperBound.branch smallRow2713_block_002_covers smallRow2713_small_right_right_covers

def smallRow2713_small : List Nat := smallRow2713_small_left ++ smallRow2713_small_right
theorem smallRow2713_small_covers : RowUpperBound PrimePairVertices.tree_small 2713 smallRow2713_small :=
  RowUpperBound.branch smallRow2713_small_left_covers smallRow2713_small_right_covers

def smallRow2713 : List Nat := smallRow2713_small
theorem smallRow2713_covers : RowUpperBound PrimePairVertices.smallTree 2713 smallRow2713 := smallRow2713_small_covers

def smallRow2719_block_000 : List Nat := []
def smallPlan2719_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2719_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2719 smallRow2719_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2719_block_000
  · decide
  · rfl

def smallRow2719_block_001 : List Nat := [3391]
def smallPlan2719_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2719_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2719 smallRow2719_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2719_block_001
  · decide
  · rfl

def smallRow2719_block_002 : List Nat := [4549, 6121]
def smallPlan2719_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .second, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow2719_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2719 smallRow2719_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2719_block_002
  · decide
  · rfl

def smallRow2719_block_003 : List Nat := [7129, 7669]
def smallPlan2719_block_003 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow2719_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2719 smallRow2719_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2719_block_003
  · decide
  · rfl

def smallRow2719_block_004 : List Nat := []
def smallPlan2719_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2719_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2719 smallRow2719_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2719_block_004
  · decide
  · rfl

def smallRow2719_small_left : List Nat := smallRow2719_block_000 ++ smallRow2719_block_001
theorem smallRow2719_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2719 smallRow2719_small_left :=
  RowUpperBound.branch smallRow2719_block_000_covers smallRow2719_block_001_covers

def smallRow2719_small_right_right : List Nat := smallRow2719_block_003 ++ smallRow2719_block_004
theorem smallRow2719_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2719 smallRow2719_small_right_right :=
  RowUpperBound.branch smallRow2719_block_003_covers smallRow2719_block_004_covers

def smallRow2719_small_right : List Nat := smallRow2719_block_002 ++ smallRow2719_small_right_right
theorem smallRow2719_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2719 smallRow2719_small_right :=
  RowUpperBound.branch smallRow2719_block_002_covers smallRow2719_small_right_right_covers

def smallRow2719_small : List Nat := smallRow2719_small_left ++ smallRow2719_small_right
theorem smallRow2719_small_covers : RowUpperBound PrimePairVertices.tree_small 2719 smallRow2719_small :=
  RowUpperBound.branch smallRow2719_small_left_covers smallRow2719_small_right_covers

def smallRow2719 : List Nat := smallRow2719_small
theorem smallRow2719_covers : RowUpperBound PrimePairVertices.smallTree 2719 smallRow2719 := smallRow2719_small_covers

def smallRow2729_block_000 : List Nat := []
def smallPlan2729_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2729_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2729 smallRow2729_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2729_block_000
  · decide
  · rfl

def smallRow2729_block_001 : List Nat := [3023, 3191]
def smallPlan2729_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2729_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2729 smallRow2729_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2729_block_001
  · decide
  · rfl

def smallRow2729_block_002 : List Nat := [4409, 5813, 6131]
def smallPlan2729_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2729_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2729 smallRow2729_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2729_block_002
  · decide
  · rfl

def smallRow2729_block_003 : List Nat := [6473, 6551, 8081]
def smallPlan2729_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2729_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2729 smallRow2729_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2729_block_003
  · decide
  · rfl

def smallRow2729_block_004 : List Nat := []
def smallPlan2729_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow2729_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2729 smallRow2729_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2729_block_004
  · decide
  · rfl

def smallRow2729_small_left : List Nat := smallRow2729_block_000 ++ smallRow2729_block_001
theorem smallRow2729_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2729 smallRow2729_small_left :=
  RowUpperBound.branch smallRow2729_block_000_covers smallRow2729_block_001_covers

def smallRow2729_small_right_right : List Nat := smallRow2729_block_003 ++ smallRow2729_block_004
theorem smallRow2729_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2729 smallRow2729_small_right_right :=
  RowUpperBound.branch smallRow2729_block_003_covers smallRow2729_block_004_covers

def smallRow2729_small_right : List Nat := smallRow2729_block_002 ++ smallRow2729_small_right_right
theorem smallRow2729_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2729 smallRow2729_small_right :=
  RowUpperBound.branch smallRow2729_block_002_covers smallRow2729_small_right_right_covers

def smallRow2729_small : List Nat := smallRow2729_small_left ++ smallRow2729_small_right
theorem smallRow2729_small_covers : RowUpperBound PrimePairVertices.tree_small 2729 smallRow2729_small :=
  RowUpperBound.branch smallRow2729_small_left_covers smallRow2729_small_right_covers

def smallRow2729 : List Nat := smallRow2729_small
theorem smallRow2729_covers : RowUpperBound PrimePairVertices.smallTree 2729 smallRow2729 := smallRow2729_small_covers

def smallRow2731_block_000 : List Nat := []
def smallPlan2731_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2731_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2731 smallRow2731_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2731_block_000
  · decide
  · rfl

def smallRow2731_block_001 : List Nat := [3301, 3571, 3613]
def smallPlan2731_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow2731_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2731 smallRow2731_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2731_block_001
  · decide
  · rfl

def smallRow2731_block_002 : List Nat := [4129, 4423, 4657, 4759, 5641]
def smallPlan2731_block_002 : List RowChoice := [.second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2731_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2731 smallRow2731_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2731_block_002
  · decide
  · rfl

def smallRow2731_block_003 : List Nat := [6451, 7753, 7867, 7993]
def smallPlan2731_block_003 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow2731_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2731 smallRow2731_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2731_block_003
  · decide
  · rfl

def smallRow2731_block_004 : List Nat := []
def smallPlan2731_block_004 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2731_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2731 smallRow2731_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2731_block_004
  · decide
  · rfl

def smallRow2731_small_left : List Nat := smallRow2731_block_000 ++ smallRow2731_block_001
theorem smallRow2731_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2731 smallRow2731_small_left :=
  RowUpperBound.branch smallRow2731_block_000_covers smallRow2731_block_001_covers

def smallRow2731_small_right_right : List Nat := smallRow2731_block_003 ++ smallRow2731_block_004
theorem smallRow2731_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2731 smallRow2731_small_right_right :=
  RowUpperBound.branch smallRow2731_block_003_covers smallRow2731_block_004_covers

def smallRow2731_small_right : List Nat := smallRow2731_block_002 ++ smallRow2731_small_right_right
theorem smallRow2731_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2731 smallRow2731_small_right :=
  RowUpperBound.branch smallRow2731_block_002_covers smallRow2731_small_right_right_covers

def smallRow2731_small : List Nat := smallRow2731_small_left ++ smallRow2731_small_right
theorem smallRow2731_small_covers : RowUpperBound PrimePairVertices.tree_small 2731 smallRow2731_small :=
  RowUpperBound.branch smallRow2731_small_left_covers smallRow2731_small_right_covers

def smallRow2731 : List Nat := smallRow2731_small
theorem smallRow2731_covers : RowUpperBound PrimePairVertices.smallTree 2731 smallRow2731 := smallRow2731_small_covers

def smallRow2741_block_000 : List Nat := []
def smallPlan2741_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2741_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2741 smallRow2741_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2741_block_000
  · decide
  · rfl

def smallRow2741_block_001 : List Nat := [2837, 2963, 3119, 3329, 3389, 3917]
def smallPlan2741_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2741_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2741 smallRow2741_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2741_block_001
  · decide
  · rfl

def smallRow2741_block_002 : List Nat := [4133, 4253, 4649, 4799, 6113]
def smallPlan2741_block_002 : List RowChoice := [.first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .second, .first, .second, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2741_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2741 smallRow2741_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2741_block_002
  · decide
  · rfl

def smallRow2741_block_003 : List Nat := [6353, 6689, 6737, 6899, 6911, 7331, 7457, 7757]
def smallPlan2741_block_003 : List RowChoice := [.first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .possible, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2741_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2741 smallRow2741_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2741_block_003
  · decide
  · rfl

def smallRow2741_block_004 : List Nat := []
def smallPlan2741_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2741_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2741 smallRow2741_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2741_block_004
  · decide
  · rfl

def smallRow2741_small_left : List Nat := smallRow2741_block_000 ++ smallRow2741_block_001
theorem smallRow2741_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2741 smallRow2741_small_left :=
  RowUpperBound.branch smallRow2741_block_000_covers smallRow2741_block_001_covers

def smallRow2741_small_right_right : List Nat := smallRow2741_block_003 ++ smallRow2741_block_004
theorem smallRow2741_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2741 smallRow2741_small_right_right :=
  RowUpperBound.branch smallRow2741_block_003_covers smallRow2741_block_004_covers

def smallRow2741_small_right : List Nat := smallRow2741_block_002 ++ smallRow2741_small_right_right
theorem smallRow2741_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2741 smallRow2741_small_right :=
  RowUpperBound.branch smallRow2741_block_002_covers smallRow2741_small_right_right_covers

def smallRow2741_small : List Nat := smallRow2741_small_left ++ smallRow2741_small_right
theorem smallRow2741_small_covers : RowUpperBound PrimePairVertices.tree_small 2741 smallRow2741_small :=
  RowUpperBound.branch smallRow2741_small_left_covers smallRow2741_small_right_covers

def smallRow2741 : List Nat := smallRow2741_small
theorem smallRow2741_covers : RowUpperBound PrimePairVertices.smallTree 2741 smallRow2741 := smallRow2741_small_covers

def smallRow2749_block_000 : List Nat := []
def smallPlan2749_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2749_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2749 smallRow2749_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2749_block_000
  · decide
  · rfl

def smallRow2749_block_001 : List Nat := [3541, 3643, 4021]
def smallPlan2749_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2749_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2749 smallRow2749_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2749_block_001
  · decide
  · rfl

def smallRow2749_block_002 : List Nat := [4261, 4297, 4993, 6067]
def smallPlan2749_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2749_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2749 smallRow2749_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2749_block_002
  · decide
  · rfl

def smallRow2749_block_003 : List Nat := [7219, 7741, 8017]
def smallPlan2749_block_003 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2749_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2749 smallRow2749_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2749_block_003
  · decide
  · rfl

def smallRow2749_block_004 : List Nat := []
def smallPlan2749_block_004 : List RowChoice := [.first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2749_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2749 smallRow2749_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2749_block_004
  · decide
  · rfl

def smallRow2749_small_left : List Nat := smallRow2749_block_000 ++ smallRow2749_block_001
theorem smallRow2749_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2749 smallRow2749_small_left :=
  RowUpperBound.branch smallRow2749_block_000_covers smallRow2749_block_001_covers

def smallRow2749_small_right_right : List Nat := smallRow2749_block_003 ++ smallRow2749_block_004
theorem smallRow2749_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2749 smallRow2749_small_right_right :=
  RowUpperBound.branch smallRow2749_block_003_covers smallRow2749_block_004_covers

def smallRow2749_small_right : List Nat := smallRow2749_block_002 ++ smallRow2749_small_right_right
theorem smallRow2749_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2749 smallRow2749_small_right :=
  RowUpperBound.branch smallRow2749_block_002_covers smallRow2749_small_right_right_covers

def smallRow2749_small : List Nat := smallRow2749_small_left ++ smallRow2749_small_right
theorem smallRow2749_small_covers : RowUpperBound PrimePairVertices.tree_small 2749 smallRow2749_small :=
  RowUpperBound.branch smallRow2749_small_left_covers smallRow2749_small_right_covers

def smallRow2749 : List Nat := smallRow2749_small
theorem smallRow2749_covers : RowUpperBound PrimePairVertices.smallTree 2749 smallRow2749 := smallRow2749_small_covers

def smallRow2753_block_000 : List Nat := []
def smallPlan2753_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2753_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 2753 smallRow2753_block_000 := by
  apply row_upper_of_plan _ _ smallPlan2753_block_000
  · decide
  · rfl

def smallRow2753_block_001 : List Nat := [2927, 2963, 3089, 3203, 3371]
def smallPlan2753_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2753_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 2753 smallRow2753_block_001 := by
  apply row_upper_of_plan _ _ smallPlan2753_block_001
  · decide
  · rfl

def smallRow2753_block_002 : List Nat := [4583, 4799, 5171, 6101]
def smallPlan2753_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2753_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 2753 smallRow2753_block_002 := by
  apply row_upper_of_plan _ _ smallPlan2753_block_002
  · decide
  · rfl

def smallRow2753_block_003 : List Nat := [6287, 6791, 7001, 7457, 7481, 7487, 7529, 7919]
def smallPlan2753_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .possible, .possible, .first, .first, .first, .first, .first, .possible, .first, .second, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow2753_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 2753 smallRow2753_block_003 := by
  apply row_upper_of_plan _ _ smallPlan2753_block_003
  · decide
  · rfl

def smallRow2753_block_004 : List Nat := []
def smallPlan2753_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow2753_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 2753 smallRow2753_block_004 := by
  apply row_upper_of_plan _ _ smallPlan2753_block_004
  · decide
  · rfl

def smallRow2753_small_left : List Nat := smallRow2753_block_000 ++ smallRow2753_block_001
theorem smallRow2753_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 2753 smallRow2753_small_left :=
  RowUpperBound.branch smallRow2753_block_000_covers smallRow2753_block_001_covers

def smallRow2753_small_right_right : List Nat := smallRow2753_block_003 ++ smallRow2753_block_004
theorem smallRow2753_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 2753 smallRow2753_small_right_right :=
  RowUpperBound.branch smallRow2753_block_003_covers smallRow2753_block_004_covers

def smallRow2753_small_right : List Nat := smallRow2753_block_002 ++ smallRow2753_small_right_right
theorem smallRow2753_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 2753 smallRow2753_small_right :=
  RowUpperBound.branch smallRow2753_block_002_covers smallRow2753_small_right_right_covers

def smallRow2753_small : List Nat := smallRow2753_small_left ++ smallRow2753_small_right
theorem smallRow2753_small_covers : RowUpperBound PrimePairVertices.tree_small 2753 smallRow2753_small :=
  RowUpperBound.branch smallRow2753_small_left_covers smallRow2753_small_right_covers

def smallRow2753 : List Nat := smallRow2753_small
theorem smallRow2753_covers : RowUpperBound PrimePairVertices.smallTree 2753 smallRow2753 := smallRow2753_small_covers

end ProjectEulerSolutions.PrimePairUpper
