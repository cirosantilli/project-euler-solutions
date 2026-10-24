import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairUpperBound

/-! Generated possible-edge rows; every excluded edge is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairUpper
open PrimePairCertificate

def smallRow3253_block_000 : List Nat := []
def smallPlan3253_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3253_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3253 smallRow3253_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3253_block_000
  · decide
  · rfl

def smallRow3253_block_001 : List Nat := [3517, 3547, 3607]
def smallPlan3253_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3253_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3253 smallRow3253_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3253_block_001
  · decide
  · rfl

def smallRow3253_block_002 : List Nat := [4219, 4231, 4339, 4357, 5197, 5479, 5557, 5851, 6043, 6067]
def smallPlan3253_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3253_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3253 smallRow3253_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3253_block_002
  · decide
  · rfl

def smallRow3253_block_003 : List Nat := [6571, 7717, 7873, 7927]
def smallPlan3253_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3253_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3253 smallRow3253_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3253_block_003
  · decide
  · rfl

def smallRow3253_block_004 : List Nat := [8263]
def smallPlan3253_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3253_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3253 smallRow3253_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3253_block_004
  · decide
  · rfl

def smallRow3253_small_left : List Nat := smallRow3253_block_000 ++ smallRow3253_block_001
theorem smallRow3253_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3253 smallRow3253_small_left :=
  RowUpperBound.branch smallRow3253_block_000_covers smallRow3253_block_001_covers

def smallRow3253_small_right_right : List Nat := smallRow3253_block_003 ++ smallRow3253_block_004
theorem smallRow3253_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3253 smallRow3253_small_right_right :=
  RowUpperBound.branch smallRow3253_block_003_covers smallRow3253_block_004_covers

def smallRow3253_small_right : List Nat := smallRow3253_block_002 ++ smallRow3253_small_right_right
theorem smallRow3253_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3253 smallRow3253_small_right :=
  RowUpperBound.branch smallRow3253_block_002_covers smallRow3253_small_right_right_covers

def smallRow3253_small : List Nat := smallRow3253_small_left ++ smallRow3253_small_right
theorem smallRow3253_small_covers : RowUpperBound PrimePairVertices.tree_small 3253 smallRow3253_small :=
  RowUpperBound.branch smallRow3253_small_left_covers smallRow3253_small_right_covers

def smallRow3253 : List Nat := smallRow3253_small
theorem smallRow3253_covers : RowUpperBound PrimePairVertices.smallTree 3253 smallRow3253 := smallRow3253_small_covers

def smallRow3257_block_000 : List Nat := []
def smallPlan3257_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3257_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3257 smallRow3257_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3257_block_000
  · decide
  · rfl

def smallRow3257_block_001 : List Nat := [3407, 3449]
def smallPlan3257_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .possible, .second, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3257_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3257 smallRow3257_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3257_block_001
  · decide
  · rfl

def smallRow3257_block_002 : List Nat := [4547, 4649, 4937, 5051, 5441, 5471, 5507, 6053]
def smallPlan3257_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3257_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3257 smallRow3257_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3257_block_002
  · decide
  · rfl

def smallRow3257_block_003 : List Nat := [6221, 6491, 6917, 7949]
def smallPlan3257_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3257_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3257 smallRow3257_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3257_block_003
  · decide
  · rfl

def smallRow3257_block_004 : List Nat := []
def smallPlan3257_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3257_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3257 smallRow3257_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3257_block_004
  · decide
  · rfl

def smallRow3257_small_left : List Nat := smallRow3257_block_000 ++ smallRow3257_block_001
theorem smallRow3257_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3257 smallRow3257_small_left :=
  RowUpperBound.branch smallRow3257_block_000_covers smallRow3257_block_001_covers

def smallRow3257_small_right_right : List Nat := smallRow3257_block_003 ++ smallRow3257_block_004
theorem smallRow3257_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3257 smallRow3257_small_right_right :=
  RowUpperBound.branch smallRow3257_block_003_covers smallRow3257_block_004_covers

def smallRow3257_small_right : List Nat := smallRow3257_block_002 ++ smallRow3257_small_right_right
theorem smallRow3257_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3257 smallRow3257_small_right :=
  RowUpperBound.branch smallRow3257_block_002_covers smallRow3257_small_right_right_covers

def smallRow3257_small : List Nat := smallRow3257_small_left ++ smallRow3257_small_right
theorem smallRow3257_small_covers : RowUpperBound PrimePairVertices.tree_small 3257 smallRow3257_small :=
  RowUpperBound.branch smallRow3257_small_left_covers smallRow3257_small_right_covers

def smallRow3257 : List Nat := smallRow3257_small
theorem smallRow3257_covers : RowUpperBound PrimePairVertices.smallTree 3257 smallRow3257 := smallRow3257_small_covers

def smallRow3259_block_000 : List Nat := []
def smallPlan3259_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3259_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3259 smallRow3259_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3259_block_000
  · decide
  · rfl

def smallRow3259_block_001 : List Nat := [3271, 4021]
def smallPlan3259_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3259_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3259 smallRow3259_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3259_block_001
  · decide
  · rfl

def smallRow3259_block_002 : List Nat := [4327, 4567, 5107, 5503, 5569]
def smallPlan3259_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3259_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3259 smallRow3259_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3259_block_002
  · decide
  · rfl

def smallRow3259_block_003 : List Nat := [6199, 6247, 7207, 7717, 7963]
def smallPlan3259_block_003 : List RowChoice := [.first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3259_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3259 smallRow3259_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3259_block_003
  · decide
  · rfl

def smallRow3259_block_004 : List Nat := []
def smallPlan3259_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3259_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3259 smallRow3259_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3259_block_004
  · decide
  · rfl

def smallRow3259_small_left : List Nat := smallRow3259_block_000 ++ smallRow3259_block_001
theorem smallRow3259_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3259 smallRow3259_small_left :=
  RowUpperBound.branch smallRow3259_block_000_covers smallRow3259_block_001_covers

def smallRow3259_small_right_right : List Nat := smallRow3259_block_003 ++ smallRow3259_block_004
theorem smallRow3259_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3259 smallRow3259_small_right_right :=
  RowUpperBound.branch smallRow3259_block_003_covers smallRow3259_block_004_covers

def smallRow3259_small_right : List Nat := smallRow3259_block_002 ++ smallRow3259_small_right_right
theorem smallRow3259_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3259 smallRow3259_small_right :=
  RowUpperBound.branch smallRow3259_block_002_covers smallRow3259_small_right_right_covers

def smallRow3259_small : List Nat := smallRow3259_small_left ++ smallRow3259_small_right
theorem smallRow3259_small_covers : RowUpperBound PrimePairVertices.tree_small 3259 smallRow3259_small :=
  RowUpperBound.branch smallRow3259_small_left_covers smallRow3259_small_right_covers

def smallRow3259 : List Nat := smallRow3259_small
theorem smallRow3259_covers : RowUpperBound PrimePairVertices.smallTree 3259 smallRow3259 := smallRow3259_small_covers

def smallRow3271_block_000 : List Nat := []
def smallPlan3271_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3271_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3271 smallRow3271_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3271_block_000
  · decide
  · rfl

def smallRow3271_block_001 : List Nat := [3319, 3361]
def smallPlan3271_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3271_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3271 smallRow3271_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3271_block_001
  · decide
  · rfl

def smallRow3271_block_002 : List Nat := [4243, 5227, 5653, 5923]
def smallPlan3271_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3271_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3271 smallRow3271_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3271_block_002
  · decide
  · rfl

def smallRow3271_block_003 : List Nat := [6661, 7603, 7621]
def smallPlan3271_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .possible, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3271_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3271 smallRow3271_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3271_block_003
  · decide
  · rfl

def smallRow3271_block_004 : List Nat := [8377]
def smallPlan3271_block_004 : List RowChoice := [.first, .first, .second, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem smallRow3271_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3271 smallRow3271_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3271_block_004
  · decide
  · rfl

def smallRow3271_small_left : List Nat := smallRow3271_block_000 ++ smallRow3271_block_001
theorem smallRow3271_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3271 smallRow3271_small_left :=
  RowUpperBound.branch smallRow3271_block_000_covers smallRow3271_block_001_covers

def smallRow3271_small_right_right : List Nat := smallRow3271_block_003 ++ smallRow3271_block_004
theorem smallRow3271_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3271 smallRow3271_small_right_right :=
  RowUpperBound.branch smallRow3271_block_003_covers smallRow3271_block_004_covers

def smallRow3271_small_right : List Nat := smallRow3271_block_002 ++ smallRow3271_small_right_right
theorem smallRow3271_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3271 smallRow3271_small_right :=
  RowUpperBound.branch smallRow3271_block_002_covers smallRow3271_small_right_right_covers

def smallRow3271_small : List Nat := smallRow3271_small_left ++ smallRow3271_small_right
theorem smallRow3271_small_covers : RowUpperBound PrimePairVertices.tree_small 3271 smallRow3271_small :=
  RowUpperBound.branch smallRow3271_small_left_covers smallRow3271_small_right_covers

def smallRow3271 : List Nat := smallRow3271_small
theorem smallRow3271_covers : RowUpperBound PrimePairVertices.smallTree 3271 smallRow3271 := smallRow3271_small_covers

def smallRow3299_block_000 : List Nat := []
def smallPlan3299_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3299_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3299 smallRow3299_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3299_block_000
  · decide
  · rfl

def smallRow3299_block_001 : List Nat := [3767, 4019]
def smallPlan3299_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3299_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3299 smallRow3299_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3299_block_001
  · decide
  · rfl

def smallRow3299_block_002 : List Nat := [4523, 4817, 6011]
def smallPlan3299_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3299_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3299 smallRow3299_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3299_block_002
  · decide
  · rfl

def smallRow3299_block_003 : List Nat := [6203, 6869, 6959, 7043, 7499]
def smallPlan3299_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3299_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3299 smallRow3299_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3299_block_003
  · decide
  · rfl

def smallRow3299_block_004 : List Nat := []
def smallPlan3299_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3299_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3299 smallRow3299_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3299_block_004
  · decide
  · rfl

def smallRow3299_small_left : List Nat := smallRow3299_block_000 ++ smallRow3299_block_001
theorem smallRow3299_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3299 smallRow3299_small_left :=
  RowUpperBound.branch smallRow3299_block_000_covers smallRow3299_block_001_covers

def smallRow3299_small_right_right : List Nat := smallRow3299_block_003 ++ smallRow3299_block_004
theorem smallRow3299_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3299 smallRow3299_small_right_right :=
  RowUpperBound.branch smallRow3299_block_003_covers smallRow3299_block_004_covers

def smallRow3299_small_right : List Nat := smallRow3299_block_002 ++ smallRow3299_small_right_right
theorem smallRow3299_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3299 smallRow3299_small_right :=
  RowUpperBound.branch smallRow3299_block_002_covers smallRow3299_small_right_right_covers

def smallRow3299_small : List Nat := smallRow3299_small_left ++ smallRow3299_small_right
theorem smallRow3299_small_covers : RowUpperBound PrimePairVertices.tree_small 3299 smallRow3299_small :=
  RowUpperBound.branch smallRow3299_small_left_covers smallRow3299_small_right_covers

def smallRow3299 : List Nat := smallRow3299_small
theorem smallRow3299_covers : RowUpperBound PrimePairVertices.smallTree 3299 smallRow3299 := smallRow3299_small_covers

def smallRow3301_block_000 : List Nat := []
def smallPlan3301_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3301_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3301 smallRow3301_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3301_block_000
  · decide
  · rfl

def smallRow3301_block_001 : List Nat := [3517]
def smallPlan3301_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3301_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3301 smallRow3301_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3301_block_001
  · decide
  · rfl

def smallRow3301_block_002 : List Nat := [4363, 4447, 4567, 4651, 5779, 5881]
def smallPlan3301_block_002 : List RowChoice := [.first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .second, .first, .first, .second, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3301_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3301 smallRow3301_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3301_block_002
  · decide
  · rfl

def smallRow3301_block_003 : List Nat := [6343, 6961, 6967, 7297]
def smallPlan3301_block_003 : List RowChoice := [.first, .second, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow3301_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3301 smallRow3301_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3301_block_003
  · decide
  · rfl

def smallRow3301_block_004 : List Nat := []
def smallPlan3301_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3301_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3301 smallRow3301_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3301_block_004
  · decide
  · rfl

def smallRow3301_small_left : List Nat := smallRow3301_block_000 ++ smallRow3301_block_001
theorem smallRow3301_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3301 smallRow3301_small_left :=
  RowUpperBound.branch smallRow3301_block_000_covers smallRow3301_block_001_covers

def smallRow3301_small_right_right : List Nat := smallRow3301_block_003 ++ smallRow3301_block_004
theorem smallRow3301_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3301 smallRow3301_small_right_right :=
  RowUpperBound.branch smallRow3301_block_003_covers smallRow3301_block_004_covers

def smallRow3301_small_right : List Nat := smallRow3301_block_002 ++ smallRow3301_small_right_right
theorem smallRow3301_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3301 smallRow3301_small_right :=
  RowUpperBound.branch smallRow3301_block_002_covers smallRow3301_small_right_right_covers

def smallRow3301_small : List Nat := smallRow3301_small_left ++ smallRow3301_small_right
theorem smallRow3301_small_covers : RowUpperBound PrimePairVertices.tree_small 3301 smallRow3301_small :=
  RowUpperBound.branch smallRow3301_small_left_covers smallRow3301_small_right_covers

def smallRow3301 : List Nat := smallRow3301_small
theorem smallRow3301_covers : RowUpperBound PrimePairVertices.smallTree 3301 smallRow3301 := smallRow3301_small_covers

def smallRow3307_block_000 : List Nat := []
def smallPlan3307_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3307_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3307 smallRow3307_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3307_block_000
  · decide
  · rfl

def smallRow3307_block_001 : List Nat := [3331, 3583]
def smallPlan3307_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3307_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3307 smallRow3307_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3307_block_001
  · decide
  · rfl

def smallRow3307_block_002 : List Nat := [4519, 5443, 5737, 5953]
def smallPlan3307_block_002 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3307_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3307 smallRow3307_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3307_block_002
  · decide
  · rfl

def smallRow3307_block_003 : List Nat := [6151, 6367, 6619, 6661, 7039, 7321, 7411]
def smallPlan3307_block_003 : List RowChoice := [.possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3307_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3307 smallRow3307_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3307_block_003
  · decide
  · rfl

def smallRow3307_block_004 : List Nat := []
def smallPlan3307_block_004 : List RowChoice := [.first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3307_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3307 smallRow3307_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3307_block_004
  · decide
  · rfl

def smallRow3307_small_left : List Nat := smallRow3307_block_000 ++ smallRow3307_block_001
theorem smallRow3307_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3307 smallRow3307_small_left :=
  RowUpperBound.branch smallRow3307_block_000_covers smallRow3307_block_001_covers

def smallRow3307_small_right_right : List Nat := smallRow3307_block_003 ++ smallRow3307_block_004
theorem smallRow3307_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3307 smallRow3307_small_right_right :=
  RowUpperBound.branch smallRow3307_block_003_covers smallRow3307_block_004_covers

def smallRow3307_small_right : List Nat := smallRow3307_block_002 ++ smallRow3307_small_right_right
theorem smallRow3307_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3307 smallRow3307_small_right :=
  RowUpperBound.branch smallRow3307_block_002_covers smallRow3307_small_right_right_covers

def smallRow3307_small : List Nat := smallRow3307_small_left ++ smallRow3307_small_right
theorem smallRow3307_small_covers : RowUpperBound PrimePairVertices.tree_small 3307 smallRow3307_small :=
  RowUpperBound.branch smallRow3307_small_left_covers smallRow3307_small_right_covers

def smallRow3307 : List Nat := smallRow3307_small
theorem smallRow3307_covers : RowUpperBound PrimePairVertices.smallTree 3307 smallRow3307 := smallRow3307_small_covers

def smallRow3313_block_000 : List Nat := []
def smallPlan3313_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3313_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3313 smallRow3313_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3313_block_000
  · decide
  · rfl

def smallRow3313_block_001 : List Nat := [3673, 3733]
def smallPlan3313_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3313_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3313 smallRow3313_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3313_block_001
  · decide
  · rfl

def smallRow3313_block_002 : List Nat := [4111, 4159, 4663, 4723, 4831, 4951]
def smallPlan3313_block_002 : List RowChoice := [.first, .possible, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3313_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3313 smallRow3313_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3313_block_002
  · decide
  · rfl

def smallRow3313_block_003 : List Nat := [6421, 6637, 6871, 8059, 8089]
def smallPlan3313_block_003 : List RowChoice := [.second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .second, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow3313_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3313 smallRow3313_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3313_block_003
  · decide
  · rfl

def smallRow3313_block_004 : List Nat := []
def smallPlan3313_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3313_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3313 smallRow3313_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3313_block_004
  · decide
  · rfl

def smallRow3313_small_left : List Nat := smallRow3313_block_000 ++ smallRow3313_block_001
theorem smallRow3313_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3313 smallRow3313_small_left :=
  RowUpperBound.branch smallRow3313_block_000_covers smallRow3313_block_001_covers

def smallRow3313_small_right_right : List Nat := smallRow3313_block_003 ++ smallRow3313_block_004
theorem smallRow3313_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3313 smallRow3313_small_right_right :=
  RowUpperBound.branch smallRow3313_block_003_covers smallRow3313_block_004_covers

def smallRow3313_small_right : List Nat := smallRow3313_block_002 ++ smallRow3313_small_right_right
theorem smallRow3313_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3313 smallRow3313_small_right :=
  RowUpperBound.branch smallRow3313_block_002_covers smallRow3313_small_right_right_covers

def smallRow3313_small : List Nat := smallRow3313_small_left ++ smallRow3313_small_right
theorem smallRow3313_small_covers : RowUpperBound PrimePairVertices.tree_small 3313 smallRow3313_small :=
  RowUpperBound.branch smallRow3313_small_left_covers smallRow3313_small_right_covers

def smallRow3313 : List Nat := smallRow3313_small
theorem smallRow3313_covers : RowUpperBound PrimePairVertices.smallTree 3313 smallRow3313 := smallRow3313_small_covers

end ProjectEulerSolutions.PrimePairUpper
