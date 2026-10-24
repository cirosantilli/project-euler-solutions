import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairUpperBound

/-! Generated possible-edge rows; every excluded edge is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairUpper
open PrimePairCertificate

def smallRow4507_block_000 : List Nat := []
def smallPlan4507_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4507_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4507 smallRow4507_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4507_block_000
  · decide
  · rfl

def smallRow4507_block_001 : List Nat := []
def smallPlan4507_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4507_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4507 smallRow4507_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4507_block_001
  · decide
  · rfl

def smallRow4507_block_002 : List Nat := [4789, 4801, 4999, 5419, 5881, 6091]
def smallPlan4507_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4507_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4507 smallRow4507_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4507_block_002
  · decide
  · rfl

def smallRow4507_block_003 : List Nat := [6217, 6709, 7129, 7603, 7759, 7993]
def smallPlan4507_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4507_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4507 smallRow4507_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4507_block_003
  · decide
  · rfl

def smallRow4507_block_004 : List Nat := [8377]
def smallPlan4507_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem smallRow4507_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4507 smallRow4507_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4507_block_004
  · decide
  · rfl

def smallRow4507_small_left : List Nat := smallRow4507_block_000 ++ smallRow4507_block_001
theorem smallRow4507_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4507 smallRow4507_small_left :=
  RowUpperBound.branch smallRow4507_block_000_covers smallRow4507_block_001_covers

def smallRow4507_small_right_right : List Nat := smallRow4507_block_003 ++ smallRow4507_block_004
theorem smallRow4507_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4507 smallRow4507_small_right_right :=
  RowUpperBound.branch smallRow4507_block_003_covers smallRow4507_block_004_covers

def smallRow4507_small_right : List Nat := smallRow4507_block_002 ++ smallRow4507_small_right_right
theorem smallRow4507_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4507 smallRow4507_small_right :=
  RowUpperBound.branch smallRow4507_block_002_covers smallRow4507_small_right_right_covers

def smallRow4507_small : List Nat := smallRow4507_small_left ++ smallRow4507_small_right
theorem smallRow4507_small_covers : RowUpperBound PrimePairVertices.tree_small 4507 smallRow4507_small :=
  RowUpperBound.branch smallRow4507_small_left_covers smallRow4507_small_right_covers

def smallRow4507 : List Nat := smallRow4507_small
theorem smallRow4507_covers : RowUpperBound PrimePairVertices.smallTree 4507 smallRow4507 := smallRow4507_small_covers

def smallRow4513_block_000 : List Nat := []
def smallPlan4513_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4513_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4513 smallRow4513_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4513_block_000
  · decide
  · rfl

def smallRow4513_block_001 : List Nat := []
def smallPlan4513_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4513_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4513 smallRow4513_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4513_block_001
  · decide
  · rfl

def smallRow4513_block_002 : List Nat := [4591, 5437, 5581]
def smallPlan4513_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4513_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4513 smallRow4513_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4513_block_002
  · decide
  · rfl

def smallRow4513_block_003 : List Nat := [6163, 6277, 6397, 6841, 7723]
def smallPlan4513_block_003 : List RowChoice := [.first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4513_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4513 smallRow4513_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4513_block_003
  · decide
  · rfl

def smallRow4513_block_004 : List Nat := [8287]
def smallPlan4513_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4513_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4513 smallRow4513_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4513_block_004
  · decide
  · rfl

def smallRow4513_small_left : List Nat := smallRow4513_block_000 ++ smallRow4513_block_001
theorem smallRow4513_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4513 smallRow4513_small_left :=
  RowUpperBound.branch smallRow4513_block_000_covers smallRow4513_block_001_covers

def smallRow4513_small_right_right : List Nat := smallRow4513_block_003 ++ smallRow4513_block_004
theorem smallRow4513_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4513 smallRow4513_small_right_right :=
  RowUpperBound.branch smallRow4513_block_003_covers smallRow4513_block_004_covers

def smallRow4513_small_right : List Nat := smallRow4513_block_002 ++ smallRow4513_small_right_right
theorem smallRow4513_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4513 smallRow4513_small_right :=
  RowUpperBound.branch smallRow4513_block_002_covers smallRow4513_small_right_right_covers

def smallRow4513_small : List Nat := smallRow4513_small_left ++ smallRow4513_small_right
theorem smallRow4513_small_covers : RowUpperBound PrimePairVertices.tree_small 4513 smallRow4513_small :=
  RowUpperBound.branch smallRow4513_small_left_covers smallRow4513_small_right_covers

def smallRow4513 : List Nat := smallRow4513_small
theorem smallRow4513_covers : RowUpperBound PrimePairVertices.smallTree 4513 smallRow4513 := smallRow4513_small_covers

def smallRow4517_block_000 : List Nat := []
def smallPlan4517_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4517_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4517 smallRow4517_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4517_block_000
  · decide
  · rfl

def smallRow4517_block_001 : List Nat := []
def smallPlan4517_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4517_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4517 smallRow4517_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4517_block_001
  · decide
  · rfl

def smallRow4517_block_002 : List Nat := [5087, 5279, 5387, 5849]
def smallPlan4517_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4517_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4517 smallRow4517_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4517_block_002
  · decide
  · rfl

def smallRow4517_block_003 : List Nat := [6581, 6689, 6869, 6959, 7523, 8087]
def smallPlan4517_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4517_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4517 smallRow4517_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4517_block_003
  · decide
  · rfl

def smallRow4517_block_004 : List Nat := [8387]
def smallPlan4517_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible]
set_option maxRecDepth 2048 in
theorem smallRow4517_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4517 smallRow4517_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4517_block_004
  · decide
  · rfl

def smallRow4517_small_left : List Nat := smallRow4517_block_000 ++ smallRow4517_block_001
theorem smallRow4517_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4517 smallRow4517_small_left :=
  RowUpperBound.branch smallRow4517_block_000_covers smallRow4517_block_001_covers

def smallRow4517_small_right_right : List Nat := smallRow4517_block_003 ++ smallRow4517_block_004
theorem smallRow4517_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4517 smallRow4517_small_right_right :=
  RowUpperBound.branch smallRow4517_block_003_covers smallRow4517_block_004_covers

def smallRow4517_small_right : List Nat := smallRow4517_block_002 ++ smallRow4517_small_right_right
theorem smallRow4517_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4517 smallRow4517_small_right :=
  RowUpperBound.branch smallRow4517_block_002_covers smallRow4517_small_right_right_covers

def smallRow4517_small : List Nat := smallRow4517_small_left ++ smallRow4517_small_right
theorem smallRow4517_small_covers : RowUpperBound PrimePairVertices.tree_small 4517 smallRow4517_small :=
  RowUpperBound.branch smallRow4517_small_left_covers smallRow4517_small_right_covers

def smallRow4517 : List Nat := smallRow4517_small
theorem smallRow4517_covers : RowUpperBound PrimePairVertices.smallTree 4517 smallRow4517 := smallRow4517_small_covers

def smallRow4519_block_000 : List Nat := []
def smallPlan4519_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4519_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4519 smallRow4519_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4519_block_000
  · decide
  · rfl

def smallRow4519_block_001 : List Nat := []
def smallPlan4519_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4519_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4519 smallRow4519_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4519_block_001
  · decide
  · rfl

def smallRow4519_block_002 : List Nat := [4657, 4729, 5209, 5419, 5701, 6043, 6133]
def smallPlan4519_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem smallRow4519_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4519 smallRow4519_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4519_block_002
  · decide
  · rfl

def smallRow4519_block_003 : List Nat := []
def smallPlan4519_block_003 : List RowChoice := [.first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .second, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4519_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4519 smallRow4519_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4519_block_003
  · decide
  · rfl

def smallRow4519_block_004 : List Nat := []
def smallPlan4519_block_004 : List RowChoice := [.first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4519_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4519 smallRow4519_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4519_block_004
  · decide
  · rfl

def smallRow4519_small_left : List Nat := smallRow4519_block_000 ++ smallRow4519_block_001
theorem smallRow4519_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4519 smallRow4519_small_left :=
  RowUpperBound.branch smallRow4519_block_000_covers smallRow4519_block_001_covers

def smallRow4519_small_right_right : List Nat := smallRow4519_block_003 ++ smallRow4519_block_004
theorem smallRow4519_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4519 smallRow4519_small_right_right :=
  RowUpperBound.branch smallRow4519_block_003_covers smallRow4519_block_004_covers

def smallRow4519_small_right : List Nat := smallRow4519_block_002 ++ smallRow4519_small_right_right
theorem smallRow4519_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4519 smallRow4519_small_right :=
  RowUpperBound.branch smallRow4519_block_002_covers smallRow4519_small_right_right_covers

def smallRow4519_small : List Nat := smallRow4519_small_left ++ smallRow4519_small_right
theorem smallRow4519_small_covers : RowUpperBound PrimePairVertices.tree_small 4519 smallRow4519_small :=
  RowUpperBound.branch smallRow4519_small_left_covers smallRow4519_small_right_covers

def smallRow4519 : List Nat := smallRow4519_small
theorem smallRow4519_covers : RowUpperBound PrimePairVertices.smallTree 4519 smallRow4519 := smallRow4519_small_covers

def smallRow4523_block_000 : List Nat := []
def smallPlan4523_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4523_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4523 smallRow4523_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4523_block_000
  · decide
  · rfl

def smallRow4523_block_001 : List Nat := []
def smallPlan4523_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4523_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4523 smallRow4523_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4523_block_001
  · decide
  · rfl

def smallRow4523_block_002 : List Nat := [4817, 4967, 5039, 5081, 5153, 5639]
def smallPlan4523_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4523_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4523 smallRow4523_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4523_block_002
  · decide
  · rfl

def smallRow4523_block_003 : List Nat := [6173, 6329, 6551, 6761, 6911, 7457, 7583, 7937, 8093]
def smallPlan4523_block_003 : List RowChoice := [.first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4523_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4523 smallRow4523_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4523_block_003
  · decide
  · rfl

def smallRow4523_block_004 : List Nat := []
def smallPlan4523_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4523_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4523 smallRow4523_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4523_block_004
  · decide
  · rfl

def smallRow4523_small_left : List Nat := smallRow4523_block_000 ++ smallRow4523_block_001
theorem smallRow4523_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4523 smallRow4523_small_left :=
  RowUpperBound.branch smallRow4523_block_000_covers smallRow4523_block_001_covers

def smallRow4523_small_right_right : List Nat := smallRow4523_block_003 ++ smallRow4523_block_004
theorem smallRow4523_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4523 smallRow4523_small_right_right :=
  RowUpperBound.branch smallRow4523_block_003_covers smallRow4523_block_004_covers

def smallRow4523_small_right : List Nat := smallRow4523_block_002 ++ smallRow4523_small_right_right
theorem smallRow4523_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4523 smallRow4523_small_right :=
  RowUpperBound.branch smallRow4523_block_002_covers smallRow4523_small_right_right_covers

def smallRow4523_small : List Nat := smallRow4523_small_left ++ smallRow4523_small_right
theorem smallRow4523_small_covers : RowUpperBound PrimePairVertices.tree_small 4523 smallRow4523_small :=
  RowUpperBound.branch smallRow4523_small_left_covers smallRow4523_small_right_covers

def smallRow4523 : List Nat := smallRow4523_small
theorem smallRow4523_covers : RowUpperBound PrimePairVertices.smallTree 4523 smallRow4523 := smallRow4523_small_covers

def smallRow4547_block_000 : List Nat := []
def smallPlan4547_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4547_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4547 smallRow4547_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4547_block_000
  · decide
  · rfl

def smallRow4547_block_001 : List Nat := []
def smallPlan4547_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4547_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4547 smallRow4547_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4547_block_001
  · decide
  · rfl

def smallRow4547_block_002 : List Nat := [4943, 4973]
def smallPlan4547_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow4547_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4547 smallRow4547_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4547_block_002
  · decide
  · rfl

def smallRow4547_block_003 : List Nat := [6311, 6521, 6689, 6791, 6863, 7283, 7547, 7919]
def smallPlan4547_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4547_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4547 smallRow4547_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4547_block_003
  · decide
  · rfl

def smallRow4547_block_004 : List Nat := []
def smallPlan4547_block_004 : List RowChoice := [.first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4547_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4547 smallRow4547_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4547_block_004
  · decide
  · rfl

def smallRow4547_small_left : List Nat := smallRow4547_block_000 ++ smallRow4547_block_001
theorem smallRow4547_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4547 smallRow4547_small_left :=
  RowUpperBound.branch smallRow4547_block_000_covers smallRow4547_block_001_covers

def smallRow4547_small_right_right : List Nat := smallRow4547_block_003 ++ smallRow4547_block_004
theorem smallRow4547_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4547 smallRow4547_small_right_right :=
  RowUpperBound.branch smallRow4547_block_003_covers smallRow4547_block_004_covers

def smallRow4547_small_right : List Nat := smallRow4547_block_002 ++ smallRow4547_small_right_right
theorem smallRow4547_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4547 smallRow4547_small_right :=
  RowUpperBound.branch smallRow4547_block_002_covers smallRow4547_small_right_right_covers

def smallRow4547_small : List Nat := smallRow4547_small_left ++ smallRow4547_small_right
theorem smallRow4547_small_covers : RowUpperBound PrimePairVertices.tree_small 4547 smallRow4547_small :=
  RowUpperBound.branch smallRow4547_small_left_covers smallRow4547_small_right_covers

def smallRow4547 : List Nat := smallRow4547_small
theorem smallRow4547_covers : RowUpperBound PrimePairVertices.smallTree 4547 smallRow4547 := smallRow4547_small_covers

def smallRow4549_block_000 : List Nat := []
def smallPlan4549_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4549_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4549 smallRow4549_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4549_block_000
  · decide
  · rfl

def smallRow4549_block_001 : List Nat := []
def smallPlan4549_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4549_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4549 smallRow4549_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4549_block_001
  · decide
  · rfl

def smallRow4549_block_002 : List Nat := [5563, 5839, 5851, 6133]
def smallPlan4549_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem smallRow4549_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4549 smallRow4549_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4549_block_002
  · decide
  · rfl

def smallRow4549_block_003 : List Nat := [6151, 6343, 6361, 6571, 7717]
def smallPlan4549_block_003 : List RowChoice := [.possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4549_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4549 smallRow4549_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4549_block_003
  · decide
  · rfl

def smallRow4549_block_004 : List Nat := [8287]
def smallPlan4549_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4549_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4549 smallRow4549_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4549_block_004
  · decide
  · rfl

def smallRow4549_small_left : List Nat := smallRow4549_block_000 ++ smallRow4549_block_001
theorem smallRow4549_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4549 smallRow4549_small_left :=
  RowUpperBound.branch smallRow4549_block_000_covers smallRow4549_block_001_covers

def smallRow4549_small_right_right : List Nat := smallRow4549_block_003 ++ smallRow4549_block_004
theorem smallRow4549_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4549 smallRow4549_small_right_right :=
  RowUpperBound.branch smallRow4549_block_003_covers smallRow4549_block_004_covers

def smallRow4549_small_right : List Nat := smallRow4549_block_002 ++ smallRow4549_small_right_right
theorem smallRow4549_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4549 smallRow4549_small_right :=
  RowUpperBound.branch smallRow4549_block_002_covers smallRow4549_small_right_right_covers

def smallRow4549_small : List Nat := smallRow4549_small_left ++ smallRow4549_small_right
theorem smallRow4549_small_covers : RowUpperBound PrimePairVertices.tree_small 4549 smallRow4549_small :=
  RowUpperBound.branch smallRow4549_small_left_covers smallRow4549_small_right_covers

def smallRow4549 : List Nat := smallRow4549_small
theorem smallRow4549_covers : RowUpperBound PrimePairVertices.smallTree 4549 smallRow4549 := smallRow4549_small_covers

def smallRow4561_block_000 : List Nat := []
def smallPlan4561_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4561_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 4561 smallRow4561_block_000 := by
  apply row_upper_of_plan _ _ smallPlan4561_block_000
  · decide
  · rfl

def smallRow4561_block_001 : List Nat := []
def smallPlan4561_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4561_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 4561 smallRow4561_block_001 := by
  apply row_upper_of_plan _ _ smallPlan4561_block_001
  · decide
  · rfl

def smallRow4561_block_002 : List Nat := [4783, 4909, 5419, 5527, 6073]
def smallPlan4561_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4561_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 4561 smallRow4561_block_002 := by
  apply row_upper_of_plan _ _ smallPlan4561_block_002
  · decide
  · rfl

def smallRow4561_block_003 : List Nat := [6379, 7723, 8017, 8101]
def smallPlan4561_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4561_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 4561 smallRow4561_block_003 := by
  apply row_upper_of_plan _ _ smallPlan4561_block_003
  · decide
  · rfl

def smallRow4561_block_004 : List Nat := []
def smallPlan4561_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow4561_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 4561 smallRow4561_block_004 := by
  apply row_upper_of_plan _ _ smallPlan4561_block_004
  · decide
  · rfl

def smallRow4561_small_left : List Nat := smallRow4561_block_000 ++ smallRow4561_block_001
theorem smallRow4561_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 4561 smallRow4561_small_left :=
  RowUpperBound.branch smallRow4561_block_000_covers smallRow4561_block_001_covers

def smallRow4561_small_right_right : List Nat := smallRow4561_block_003 ++ smallRow4561_block_004
theorem smallRow4561_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 4561 smallRow4561_small_right_right :=
  RowUpperBound.branch smallRow4561_block_003_covers smallRow4561_block_004_covers

def smallRow4561_small_right : List Nat := smallRow4561_block_002 ++ smallRow4561_small_right_right
theorem smallRow4561_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 4561 smallRow4561_small_right :=
  RowUpperBound.branch smallRow4561_block_002_covers smallRow4561_small_right_right_covers

def smallRow4561_small : List Nat := smallRow4561_small_left ++ smallRow4561_small_right
theorem smallRow4561_small_covers : RowUpperBound PrimePairVertices.tree_small 4561 smallRow4561_small :=
  RowUpperBound.branch smallRow4561_small_left_covers smallRow4561_small_right_covers

def smallRow4561 : List Nat := smallRow4561_small
theorem smallRow4561_covers : RowUpperBound PrimePairVertices.smallTree 4561 smallRow4561 := smallRow4561_small_covers

end ProjectEulerSolutions.PrimePairUpper
