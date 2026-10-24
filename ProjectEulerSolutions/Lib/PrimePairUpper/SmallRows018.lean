import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairUpperBound

/-! Generated possible-edge rows; every excluded edge is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairUpper
open PrimePairCertificate

def smallRow3823_block_000 : List Nat := []
def smallPlan3823_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3823_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3823 smallRow3823_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3823_block_000
  · decide
  · rfl

def smallRow3823_block_001 : List Nat := []
def smallPlan3823_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3823_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3823 smallRow3823_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3823_block_001
  · decide
  · rfl

def smallRow3823_block_002 : List Nat := [4159, 4201, 4297, 4549, 4759, 5167, 5209, 5233, 5779]
def smallPlan3823_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3823_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3823 smallRow3823_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3823_block_002
  · decide
  · rfl

def smallRow3823_block_003 : List Nat := [6199, 6841, 7393, 7417, 7993]
def smallPlan3823_block_003 : List RowChoice := [.first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .second, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3823_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3823 smallRow3823_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3823_block_003
  · decide
  · rfl

def smallRow3823_block_004 : List Nat := []
def smallPlan3823_block_004 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3823_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3823 smallRow3823_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3823_block_004
  · decide
  · rfl

def smallRow3823_small_left : List Nat := smallRow3823_block_000 ++ smallRow3823_block_001
theorem smallRow3823_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3823 smallRow3823_small_left :=
  RowUpperBound.branch smallRow3823_block_000_covers smallRow3823_block_001_covers

def smallRow3823_small_right_right : List Nat := smallRow3823_block_003 ++ smallRow3823_block_004
theorem smallRow3823_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3823 smallRow3823_small_right_right :=
  RowUpperBound.branch smallRow3823_block_003_covers smallRow3823_block_004_covers

def smallRow3823_small_right : List Nat := smallRow3823_block_002 ++ smallRow3823_small_right_right
theorem smallRow3823_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3823 smallRow3823_small_right :=
  RowUpperBound.branch smallRow3823_block_002_covers smallRow3823_small_right_right_covers

def smallRow3823_small : List Nat := smallRow3823_small_left ++ smallRow3823_small_right
theorem smallRow3823_small_covers : RowUpperBound PrimePairVertices.tree_small 3823 smallRow3823_small :=
  RowUpperBound.branch smallRow3823_small_left_covers smallRow3823_small_right_covers

def smallRow3823 : List Nat := smallRow3823_small
theorem smallRow3823_covers : RowUpperBound PrimePairVertices.smallTree 3823 smallRow3823 := smallRow3823_small_covers

def smallRow3833_block_000 : List Nat := []
def smallPlan3833_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3833_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3833 smallRow3833_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3833_block_000
  · decide
  · rfl

def smallRow3833_block_001 : List Nat := []
def smallPlan3833_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3833_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3833 smallRow3833_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3833_block_001
  · decide
  · rfl

def smallRow3833_block_002 : List Nat := [4253, 4733, 4889, 5657, 5813]
def smallPlan3833_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3833_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3833 smallRow3833_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3833_block_002
  · decide
  · rfl

def smallRow3833_block_003 : List Nat := [6269, 6449, 6827, 6833, 6911, 6983, 7841]
def smallPlan3833_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3833_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3833 smallRow3833_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3833_block_003
  · decide
  · rfl

def smallRow3833_block_004 : List Nat := []
def smallPlan3833_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3833_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3833 smallRow3833_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3833_block_004
  · decide
  · rfl

def smallRow3833_small_left : List Nat := smallRow3833_block_000 ++ smallRow3833_block_001
theorem smallRow3833_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3833 smallRow3833_small_left :=
  RowUpperBound.branch smallRow3833_block_000_covers smallRow3833_block_001_covers

def smallRow3833_small_right_right : List Nat := smallRow3833_block_003 ++ smallRow3833_block_004
theorem smallRow3833_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3833 smallRow3833_small_right_right :=
  RowUpperBound.branch smallRow3833_block_003_covers smallRow3833_block_004_covers

def smallRow3833_small_right : List Nat := smallRow3833_block_002 ++ smallRow3833_small_right_right
theorem smallRow3833_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3833 smallRow3833_small_right :=
  RowUpperBound.branch smallRow3833_block_002_covers smallRow3833_small_right_right_covers

def smallRow3833_small : List Nat := smallRow3833_small_left ++ smallRow3833_small_right
theorem smallRow3833_small_covers : RowUpperBound PrimePairVertices.tree_small 3833 smallRow3833_small :=
  RowUpperBound.branch smallRow3833_small_left_covers smallRow3833_small_right_covers

def smallRow3833 : List Nat := smallRow3833_small
theorem smallRow3833_covers : RowUpperBound PrimePairVertices.smallTree 3833 smallRow3833 := smallRow3833_small_covers

def smallRow3847_block_000 : List Nat := []
def smallPlan3847_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3847_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3847 smallRow3847_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3847_block_000
  · decide
  · rfl

def smallRow3847_block_001 : List Nat := [3931, 4021]
def smallPlan3847_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3847_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3847 smallRow3847_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3847_block_001
  · decide
  · rfl

def smallRow3847_block_002 : List Nat := [4099, 5077, 5197, 5659, 5737]
def smallPlan3847_block_002 : List RowChoice := [.possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .second, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3847_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3847 smallRow3847_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3847_block_002
  · decide
  · rfl

def smallRow3847_block_003 : List Nat := [6211, 6367, 6553, 6883]
def smallPlan3847_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3847_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3847 smallRow3847_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3847_block_003
  · decide
  · rfl

def smallRow3847_block_004 : List Nat := [8221, 8233]
def smallPlan3847_block_004 : List RowChoice := [.first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3847_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3847 smallRow3847_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3847_block_004
  · decide
  · rfl

def smallRow3847_small_left : List Nat := smallRow3847_block_000 ++ smallRow3847_block_001
theorem smallRow3847_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3847 smallRow3847_small_left :=
  RowUpperBound.branch smallRow3847_block_000_covers smallRow3847_block_001_covers

def smallRow3847_small_right_right : List Nat := smallRow3847_block_003 ++ smallRow3847_block_004
theorem smallRow3847_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3847 smallRow3847_small_right_right :=
  RowUpperBound.branch smallRow3847_block_003_covers smallRow3847_block_004_covers

def smallRow3847_small_right : List Nat := smallRow3847_block_002 ++ smallRow3847_small_right_right
theorem smallRow3847_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3847 smallRow3847_small_right :=
  RowUpperBound.branch smallRow3847_block_002_covers smallRow3847_small_right_right_covers

def smallRow3847_small : List Nat := smallRow3847_small_left ++ smallRow3847_small_right
theorem smallRow3847_small_covers : RowUpperBound PrimePairVertices.tree_small 3847 smallRow3847_small :=
  RowUpperBound.branch smallRow3847_small_left_covers smallRow3847_small_right_covers

def smallRow3847 : List Nat := smallRow3847_small
theorem smallRow3847_covers : RowUpperBound PrimePairVertices.smallTree 3847 smallRow3847 := smallRow3847_small_covers

def smallRow3851_block_000 : List Nat := []
def smallPlan3851_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3851_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3851 smallRow3851_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3851_block_000
  · decide
  · rfl

def smallRow3851_block_001 : List Nat := [4013]
def smallPlan3851_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3851_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3851 smallRow3851_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3851_block_001
  · decide
  · rfl

def smallRow3851_block_002 : List Nat := [4139, 4271, 4649, 5651, 5903]
def smallPlan3851_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3851_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3851 smallRow3851_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3851_block_002
  · decide
  · rfl

def smallRow3851_block_003 : List Nat := [6521, 6659, 6899, 6983, 7043, 7151, 7547, 7673]
def smallPlan3851_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .second, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3851_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3851 smallRow3851_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3851_block_003
  · decide
  · rfl

def smallRow3851_block_004 : List Nat := []
def smallPlan3851_block_004 : List RowChoice := [.first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow3851_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3851 smallRow3851_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3851_block_004
  · decide
  · rfl

def smallRow3851_small_left : List Nat := smallRow3851_block_000 ++ smallRow3851_block_001
theorem smallRow3851_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3851 smallRow3851_small_left :=
  RowUpperBound.branch smallRow3851_block_000_covers smallRow3851_block_001_covers

def smallRow3851_small_right_right : List Nat := smallRow3851_block_003 ++ smallRow3851_block_004
theorem smallRow3851_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3851 smallRow3851_small_right_right :=
  RowUpperBound.branch smallRow3851_block_003_covers smallRow3851_block_004_covers

def smallRow3851_small_right : List Nat := smallRow3851_block_002 ++ smallRow3851_small_right_right
theorem smallRow3851_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3851 smallRow3851_small_right :=
  RowUpperBound.branch smallRow3851_block_002_covers smallRow3851_small_right_right_covers

def smallRow3851_small : List Nat := smallRow3851_small_left ++ smallRow3851_small_right
theorem smallRow3851_small_covers : RowUpperBound PrimePairVertices.tree_small 3851 smallRow3851_small :=
  RowUpperBound.branch smallRow3851_small_left_covers smallRow3851_small_right_covers

def smallRow3851 : List Nat := smallRow3851_small
theorem smallRow3851_covers : RowUpperBound PrimePairVertices.smallTree 3851 smallRow3851 := smallRow3851_small_covers

def smallRow3853_block_000 : List Nat := []
def smallPlan3853_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3853_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3853 smallRow3853_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3853_block_000
  · decide
  · rfl

def smallRow3853_block_001 : List Nat := []
def smallPlan3853_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3853_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3853 smallRow3853_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3853_block_001
  · decide
  · rfl

def smallRow3853_block_002 : List Nat := [4099, 4759, 5521, 5683]
def smallPlan3853_block_002 : List RowChoice := [.possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3853_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3853 smallRow3853_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3853_block_002
  · decide
  · rfl

def smallRow3853_block_003 : List Nat := [6361, 6367, 6607, 6709, 6991, 7237, 7243, 7297, 7591, 7699]
def smallPlan3853_block_003 : List RowChoice := [.first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .possible, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3853_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3853 smallRow3853_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3853_block_003
  · decide
  · rfl

def smallRow3853_block_004 : List Nat := [8293]
def smallPlan3853_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3853_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3853 smallRow3853_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3853_block_004
  · decide
  · rfl

def smallRow3853_small_left : List Nat := smallRow3853_block_000 ++ smallRow3853_block_001
theorem smallRow3853_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3853 smallRow3853_small_left :=
  RowUpperBound.branch smallRow3853_block_000_covers smallRow3853_block_001_covers

def smallRow3853_small_right_right : List Nat := smallRow3853_block_003 ++ smallRow3853_block_004
theorem smallRow3853_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3853 smallRow3853_small_right_right :=
  RowUpperBound.branch smallRow3853_block_003_covers smallRow3853_block_004_covers

def smallRow3853_small_right : List Nat := smallRow3853_block_002 ++ smallRow3853_small_right_right
theorem smallRow3853_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3853 smallRow3853_small_right :=
  RowUpperBound.branch smallRow3853_block_002_covers smallRow3853_small_right_right_covers

def smallRow3853_small : List Nat := smallRow3853_small_left ++ smallRow3853_small_right
theorem smallRow3853_small_covers : RowUpperBound PrimePairVertices.tree_small 3853 smallRow3853_small :=
  RowUpperBound.branch smallRow3853_small_left_covers smallRow3853_small_right_covers

def smallRow3853 : List Nat := smallRow3853_small
theorem smallRow3853_covers : RowUpperBound PrimePairVertices.smallTree 3853 smallRow3853 := smallRow3853_small_covers

def smallRow3863_block_000 : List Nat := []
def smallPlan3863_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3863_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3863 smallRow3863_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3863_block_000
  · decide
  · rfl

def smallRow3863_block_001 : List Nat := [4007]
def smallPlan3863_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3863_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3863 smallRow3863_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3863_block_001
  · decide
  · rfl

def smallRow3863_block_002 : List Nat := [4451, 5297]
def smallPlan3863_block_002 : List RowChoice := [.first, .first, .first, .first, .second, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3863_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3863 smallRow3863_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3863_block_002
  · decide
  · rfl

def smallRow3863_block_003 : List Nat := [6569, 6779, 7187, 7673, 7757]
def smallPlan3863_block_003 : List RowChoice := [.first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3863_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3863 smallRow3863_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3863_block_003
  · decide
  · rfl

def smallRow3863_block_004 : List Nat := []
def smallPlan3863_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow3863_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3863 smallRow3863_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3863_block_004
  · decide
  · rfl

def smallRow3863_small_left : List Nat := smallRow3863_block_000 ++ smallRow3863_block_001
theorem smallRow3863_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3863 smallRow3863_small_left :=
  RowUpperBound.branch smallRow3863_block_000_covers smallRow3863_block_001_covers

def smallRow3863_small_right_right : List Nat := smallRow3863_block_003 ++ smallRow3863_block_004
theorem smallRow3863_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3863 smallRow3863_small_right_right :=
  RowUpperBound.branch smallRow3863_block_003_covers smallRow3863_block_004_covers

def smallRow3863_small_right : List Nat := smallRow3863_block_002 ++ smallRow3863_small_right_right
theorem smallRow3863_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3863 smallRow3863_small_right :=
  RowUpperBound.branch smallRow3863_block_002_covers smallRow3863_small_right_right_covers

def smallRow3863_small : List Nat := smallRow3863_small_left ++ smallRow3863_small_right
theorem smallRow3863_small_covers : RowUpperBound PrimePairVertices.tree_small 3863 smallRow3863_small :=
  RowUpperBound.branch smallRow3863_small_left_covers smallRow3863_small_right_covers

def smallRow3863 : List Nat := smallRow3863_small
theorem smallRow3863_covers : RowUpperBound PrimePairVertices.smallTree 3863 smallRow3863 := smallRow3863_small_covers

def smallRow3877_block_000 : List Nat := []
def smallPlan3877_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3877_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3877 smallRow3877_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3877_block_000
  · decide
  · rfl

def smallRow3877_block_001 : List Nat := []
def smallPlan3877_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3877_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3877 smallRow3877_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3877_block_001
  · decide
  · rfl

def smallRow3877_block_002 : List Nat := [4507, 5167, 5227, 5479, 5683]
def smallPlan3877_block_002 : List RowChoice := [.first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3877_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3877 smallRow3877_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3877_block_002
  · decide
  · rfl

def smallRow3877_block_003 : List Nat := [6151, 6163, 6373, 6469, 6607, 7159]
def smallPlan3877_block_003 : List RowChoice := [.possible, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .second, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3877_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3877 smallRow3877_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3877_block_003
  · decide
  · rfl

def smallRow3877_block_004 : List Nat := [8263]
def smallPlan3877_block_004 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .second, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3877_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3877 smallRow3877_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3877_block_004
  · decide
  · rfl

def smallRow3877_small_left : List Nat := smallRow3877_block_000 ++ smallRow3877_block_001
theorem smallRow3877_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3877 smallRow3877_small_left :=
  RowUpperBound.branch smallRow3877_block_000_covers smallRow3877_block_001_covers

def smallRow3877_small_right_right : List Nat := smallRow3877_block_003 ++ smallRow3877_block_004
theorem smallRow3877_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3877 smallRow3877_small_right_right :=
  RowUpperBound.branch smallRow3877_block_003_covers smallRow3877_block_004_covers

def smallRow3877_small_right : List Nat := smallRow3877_block_002 ++ smallRow3877_small_right_right
theorem smallRow3877_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3877 smallRow3877_small_right :=
  RowUpperBound.branch smallRow3877_block_002_covers smallRow3877_small_right_right_covers

def smallRow3877_small : List Nat := smallRow3877_small_left ++ smallRow3877_small_right
theorem smallRow3877_small_covers : RowUpperBound PrimePairVertices.tree_small 3877 smallRow3877_small :=
  RowUpperBound.branch smallRow3877_small_left_covers smallRow3877_small_right_covers

def smallRow3877 : List Nat := smallRow3877_small
theorem smallRow3877_covers : RowUpperBound PrimePairVertices.smallTree 3877 smallRow3877 := smallRow3877_small_covers

def smallRow3881_block_000 : List Nat := []
def smallPlan3881_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3881_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3881 smallRow3881_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3881_block_000
  · decide
  · rfl

def smallRow3881_block_001 : List Nat := []
def smallPlan3881_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3881_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3881 smallRow3881_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3881_block_001
  · decide
  · rfl

def smallRow3881_block_002 : List Nat := [4421, 4793, 5261, 5519]
def smallPlan3881_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3881_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3881 smallRow3881_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3881_block_002
  · decide
  · rfl

def smallRow3881_block_003 : List Nat := [6269, 6857]
def smallPlan3881_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3881_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3881 smallRow3881_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3881_block_003
  · decide
  · rfl

def smallRow3881_block_004 : List Nat := [8363, 8369]
def smallPlan3881_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3881_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3881 smallRow3881_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3881_block_004
  · decide
  · rfl

def smallRow3881_small_left : List Nat := smallRow3881_block_000 ++ smallRow3881_block_001
theorem smallRow3881_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3881 smallRow3881_small_left :=
  RowUpperBound.branch smallRow3881_block_000_covers smallRow3881_block_001_covers

def smallRow3881_small_right_right : List Nat := smallRow3881_block_003 ++ smallRow3881_block_004
theorem smallRow3881_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3881 smallRow3881_small_right_right :=
  RowUpperBound.branch smallRow3881_block_003_covers smallRow3881_block_004_covers

def smallRow3881_small_right : List Nat := smallRow3881_block_002 ++ smallRow3881_small_right_right
theorem smallRow3881_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3881 smallRow3881_small_right :=
  RowUpperBound.branch smallRow3881_block_002_covers smallRow3881_small_right_right_covers

def smallRow3881_small : List Nat := smallRow3881_small_left ++ smallRow3881_small_right
theorem smallRow3881_small_covers : RowUpperBound PrimePairVertices.tree_small 3881 smallRow3881_small :=
  RowUpperBound.branch smallRow3881_small_left_covers smallRow3881_small_right_covers

def smallRow3881 : List Nat := smallRow3881_small
theorem smallRow3881_covers : RowUpperBound PrimePairVertices.smallTree 3881 smallRow3881 := smallRow3881_small_covers

end ProjectEulerSolutions.PrimePairUpper
