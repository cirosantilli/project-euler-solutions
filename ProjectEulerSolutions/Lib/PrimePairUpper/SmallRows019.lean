import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairUpperBound

/-! Generated possible-edge rows; every excluded edge is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairUpper
open PrimePairCertificate

def smallRow3889_block_000 : List Nat := []
def smallPlan3889_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3889_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3889 smallRow3889_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3889_block_000
  · decide
  · rfl

def smallRow3889_block_001 : List Nat := [4021]
def smallPlan3889_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3889_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3889 smallRow3889_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3889_block_001
  · decide
  · rfl

def smallRow3889_block_002 : List Nat := [5749, 5821, 6007]
def smallPlan3889_block_002 : List RowChoice := [.first, .second, .first, .second, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .second, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3889_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3889 smallRow3889_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3889_block_002
  · decide
  · rfl

def smallRow3889_block_003 : List Nat := [6619, 6763, 7477, 7927]
def smallPlan3889_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .second, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .second, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3889_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3889 smallRow3889_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3889_block_003
  · decide
  · rfl

def smallRow3889_block_004 : List Nat := []
def smallPlan3889_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3889_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3889 smallRow3889_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3889_block_004
  · decide
  · rfl

def smallRow3889_small_left : List Nat := smallRow3889_block_000 ++ smallRow3889_block_001
theorem smallRow3889_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3889 smallRow3889_small_left :=
  RowUpperBound.branch smallRow3889_block_000_covers smallRow3889_block_001_covers

def smallRow3889_small_right_right : List Nat := smallRow3889_block_003 ++ smallRow3889_block_004
theorem smallRow3889_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3889 smallRow3889_small_right_right :=
  RowUpperBound.branch smallRow3889_block_003_covers smallRow3889_block_004_covers

def smallRow3889_small_right : List Nat := smallRow3889_block_002 ++ smallRow3889_small_right_right
theorem smallRow3889_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3889 smallRow3889_small_right :=
  RowUpperBound.branch smallRow3889_block_002_covers smallRow3889_small_right_right_covers

def smallRow3889_small : List Nat := smallRow3889_small_left ++ smallRow3889_small_right
theorem smallRow3889_small_covers : RowUpperBound PrimePairVertices.tree_small 3889 smallRow3889_small :=
  RowUpperBound.branch smallRow3889_small_left_covers smallRow3889_small_right_covers

def smallRow3889 : List Nat := smallRow3889_small
theorem smallRow3889_covers : RowUpperBound PrimePairVertices.smallTree 3889 smallRow3889 := smallRow3889_small_covers

def smallRow3907_block_000 : List Nat := []
def smallPlan3907_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3907_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3907 smallRow3907_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3907_block_000
  · decide
  · rfl

def smallRow3907_block_001 : List Nat := [4027]
def smallPlan3907_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3907_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3907 smallRow3907_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3907_block_001
  · decide
  · rfl

def smallRow3907_block_002 : List Nat := [4327, 4603, 5431, 6043]
def smallPlan3907_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3907_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3907 smallRow3907_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3907_block_002
  · decide
  · rfl

def smallRow3907_block_003 : List Nat := [6469, 6871, 6997, 7333, 7573]
def smallPlan3907_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3907_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3907 smallRow3907_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3907_block_003
  · decide
  · rfl

def smallRow3907_block_004 : List Nat := []
def smallPlan3907_block_004 : List RowChoice := [.first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3907_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3907 smallRow3907_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3907_block_004
  · decide
  · rfl

def smallRow3907_small_left : List Nat := smallRow3907_block_000 ++ smallRow3907_block_001
theorem smallRow3907_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3907 smallRow3907_small_left :=
  RowUpperBound.branch smallRow3907_block_000_covers smallRow3907_block_001_covers

def smallRow3907_small_right_right : List Nat := smallRow3907_block_003 ++ smallRow3907_block_004
theorem smallRow3907_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3907 smallRow3907_small_right_right :=
  RowUpperBound.branch smallRow3907_block_003_covers smallRow3907_block_004_covers

def smallRow3907_small_right : List Nat := smallRow3907_block_002 ++ smallRow3907_small_right_right
theorem smallRow3907_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3907 smallRow3907_small_right :=
  RowUpperBound.branch smallRow3907_block_002_covers smallRow3907_small_right_right_covers

def smallRow3907_small : List Nat := smallRow3907_small_left ++ smallRow3907_small_right
theorem smallRow3907_small_covers : RowUpperBound PrimePairVertices.tree_small 3907 smallRow3907_small :=
  RowUpperBound.branch smallRow3907_small_left_covers smallRow3907_small_right_covers

def smallRow3907 : List Nat := smallRow3907_small
theorem smallRow3907_covers : RowUpperBound PrimePairVertices.smallTree 3907 smallRow3907 := smallRow3907_small_covers

def smallRow3911_block_000 : List Nat := []
def smallPlan3911_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3911_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3911 smallRow3911_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3911_block_000
  · decide
  · rfl

def smallRow3911_block_001 : List Nat := [4079]
def smallPlan3911_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .possible, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3911_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3911 smallRow3911_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3911_block_001
  · decide
  · rfl

def smallRow3911_block_002 : List Nat := [4259, 4457, 5477, 5717]
def smallPlan3911_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3911_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3911 smallRow3911_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3911_block_002
  · decide
  · rfl

def smallRow3911_block_003 : List Nat := [6257, 7349, 7919, 7937]
def smallPlan3911_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3911_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3911 smallRow3911_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3911_block_003
  · decide
  · rfl

def smallRow3911_block_004 : List Nat := []
def smallPlan3911_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3911_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3911 smallRow3911_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3911_block_004
  · decide
  · rfl

def smallRow3911_small_left : List Nat := smallRow3911_block_000 ++ smallRow3911_block_001
theorem smallRow3911_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3911 smallRow3911_small_left :=
  RowUpperBound.branch smallRow3911_block_000_covers smallRow3911_block_001_covers

def smallRow3911_small_right_right : List Nat := smallRow3911_block_003 ++ smallRow3911_block_004
theorem smallRow3911_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3911 smallRow3911_small_right_right :=
  RowUpperBound.branch smallRow3911_block_003_covers smallRow3911_block_004_covers

def smallRow3911_small_right : List Nat := smallRow3911_block_002 ++ smallRow3911_small_right_right
theorem smallRow3911_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3911 smallRow3911_small_right :=
  RowUpperBound.branch smallRow3911_block_002_covers smallRow3911_small_right_right_covers

def smallRow3911_small : List Nat := smallRow3911_small_left ++ smallRow3911_small_right
theorem smallRow3911_small_covers : RowUpperBound PrimePairVertices.tree_small 3911 smallRow3911_small :=
  RowUpperBound.branch smallRow3911_small_left_covers smallRow3911_small_right_covers

def smallRow3911 : List Nat := smallRow3911_small
theorem smallRow3911_covers : RowUpperBound PrimePairVertices.smallTree 3911 smallRow3911 := smallRow3911_small_covers

def smallRow3917_block_000 : List Nat := []
def smallPlan3917_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3917_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3917 smallRow3917_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3917_block_000
  · decide
  · rfl

def smallRow3917_block_001 : List Nat := [4019]
def smallPlan3917_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3917_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3917 smallRow3917_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3917_block_001
  · decide
  · rfl

def smallRow3917_block_002 : List Nat := [4211, 4271, 4337, 4349, 4649, 5189, 5399, 5483, 5987]
def smallPlan3917_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .second, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3917_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3917 smallRow3917_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3917_block_002
  · decide
  · rfl

def smallRow3917_block_003 : List Nat := [6323, 6779, 6899, 7331, 7499]
def smallPlan3917_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .second, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3917_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3917 smallRow3917_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3917_block_003
  · decide
  · rfl

def smallRow3917_block_004 : List Nat := []
def smallPlan3917_block_004 : List RowChoice := [.first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3917_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3917 smallRow3917_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3917_block_004
  · decide
  · rfl

def smallRow3917_small_left : List Nat := smallRow3917_block_000 ++ smallRow3917_block_001
theorem smallRow3917_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3917 smallRow3917_small_left :=
  RowUpperBound.branch smallRow3917_block_000_covers smallRow3917_block_001_covers

def smallRow3917_small_right_right : List Nat := smallRow3917_block_003 ++ smallRow3917_block_004
theorem smallRow3917_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3917 smallRow3917_small_right_right :=
  RowUpperBound.branch smallRow3917_block_003_covers smallRow3917_block_004_covers

def smallRow3917_small_right : List Nat := smallRow3917_block_002 ++ smallRow3917_small_right_right
theorem smallRow3917_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3917 smallRow3917_small_right :=
  RowUpperBound.branch smallRow3917_block_002_covers smallRow3917_small_right_right_covers

def smallRow3917_small : List Nat := smallRow3917_small_left ++ smallRow3917_small_right
theorem smallRow3917_small_covers : RowUpperBound PrimePairVertices.tree_small 3917 smallRow3917_small :=
  RowUpperBound.branch smallRow3917_small_left_covers smallRow3917_small_right_covers

def smallRow3917 : List Nat := smallRow3917_small
theorem smallRow3917_covers : RowUpperBound PrimePairVertices.smallTree 3917 smallRow3917 := smallRow3917_small_covers

def smallRow3919_block_000 : List Nat := []
def smallPlan3919_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3919_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3919 smallRow3919_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3919_block_000
  · decide
  · rfl

def smallRow3919_block_001 : List Nat := []
def smallPlan3919_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3919_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3919 smallRow3919_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3919_block_001
  · decide
  · rfl

def smallRow3919_block_002 : List Nat := [4297, 4567, 4591, 4957, 5179, 5281, 5503, 5743, 6007]
def smallPlan3919_block_002 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3919_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3919 smallRow3919_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3919_block_002
  · decide
  · rfl

def smallRow3919_block_003 : List Nat := [6151, 6397, 6529, 6961, 7309, 7489, 7789, 8011]
def smallPlan3919_block_003 : List RowChoice := [.possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3919_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3919 smallRow3919_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3919_block_003
  · decide
  · rfl

def smallRow3919_block_004 : List Nat := [8317]
def smallPlan3919_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3919_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3919 smallRow3919_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3919_block_004
  · decide
  · rfl

def smallRow3919_small_left : List Nat := smallRow3919_block_000 ++ smallRow3919_block_001
theorem smallRow3919_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3919 smallRow3919_small_left :=
  RowUpperBound.branch smallRow3919_block_000_covers smallRow3919_block_001_covers

def smallRow3919_small_right_right : List Nat := smallRow3919_block_003 ++ smallRow3919_block_004
theorem smallRow3919_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3919 smallRow3919_small_right_right :=
  RowUpperBound.branch smallRow3919_block_003_covers smallRow3919_block_004_covers

def smallRow3919_small_right : List Nat := smallRow3919_block_002 ++ smallRow3919_small_right_right
theorem smallRow3919_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3919 smallRow3919_small_right :=
  RowUpperBound.branch smallRow3919_block_002_covers smallRow3919_small_right_right_covers

def smallRow3919_small : List Nat := smallRow3919_small_left ++ smallRow3919_small_right
theorem smallRow3919_small_covers : RowUpperBound PrimePairVertices.tree_small 3919 smallRow3919_small :=
  RowUpperBound.branch smallRow3919_small_left_covers smallRow3919_small_right_covers

def smallRow3919 : List Nat := smallRow3919_small
theorem smallRow3919_covers : RowUpperBound PrimePairVertices.smallTree 3919 smallRow3919 := smallRow3919_small_covers

def smallRow3923_block_000 : List Nat := []
def smallPlan3923_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3923_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3923 smallRow3923_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3923_block_000
  · decide
  · rfl

def smallRow3923_block_001 : List Nat := [3989]
def smallPlan3923_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3923_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3923 smallRow3923_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3923_block_001
  · decide
  · rfl

def smallRow3923_block_002 : List Nat := [4133, 4211, 4703, 4721, 4919, 4973, 5519, 6011]
def smallPlan3923_block_002 : List RowChoice := [.first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .possible, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem smallRow3923_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3923 smallRow3923_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3923_block_002
  · decide
  · rfl

def smallRow3923_block_003 : List Nat := [6803, 7643, 7949]
def smallPlan3923_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3923_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3923 smallRow3923_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3923_block_003
  · decide
  · rfl

def smallRow3923_block_004 : List Nat := [8237]
def smallPlan3923_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3923_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3923 smallRow3923_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3923_block_004
  · decide
  · rfl

def smallRow3923_small_left : List Nat := smallRow3923_block_000 ++ smallRow3923_block_001
theorem smallRow3923_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3923 smallRow3923_small_left :=
  RowUpperBound.branch smallRow3923_block_000_covers smallRow3923_block_001_covers

def smallRow3923_small_right_right : List Nat := smallRow3923_block_003 ++ smallRow3923_block_004
theorem smallRow3923_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3923 smallRow3923_small_right_right :=
  RowUpperBound.branch smallRow3923_block_003_covers smallRow3923_block_004_covers

def smallRow3923_small_right : List Nat := smallRow3923_block_002 ++ smallRow3923_small_right_right
theorem smallRow3923_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3923 smallRow3923_small_right :=
  RowUpperBound.branch smallRow3923_block_002_covers smallRow3923_small_right_right_covers

def smallRow3923_small : List Nat := smallRow3923_small_left ++ smallRow3923_small_right
theorem smallRow3923_small_covers : RowUpperBound PrimePairVertices.tree_small 3923 smallRow3923_small :=
  RowUpperBound.branch smallRow3923_small_left_covers smallRow3923_small_right_covers

def smallRow3923 : List Nat := smallRow3923_small
theorem smallRow3923_covers : RowUpperBound PrimePairVertices.smallTree 3923 smallRow3923 := smallRow3923_small_covers

def smallRow3929_block_000 : List Nat := []
def smallPlan3929_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3929_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3929 smallRow3929_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3929_block_000
  · decide
  · rfl

def smallRow3929_block_001 : List Nat := []
def smallPlan3929_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem smallRow3929_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3929 smallRow3929_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3929_block_001
  · decide
  · rfl

def smallRow3929_block_002 : List Nat := [4349, 4421, 6053, 6143]
def smallPlan3929_block_002 : List RowChoice := [.first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible]
set_option maxRecDepth 2048 in
theorem smallRow3929_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3929 smallRow3929_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3929_block_002
  · decide
  · rfl

def smallRow3929_block_003 : List Nat := [6221, 7043, 7109, 7457, 7691]
def smallPlan3929_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3929_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3929 smallRow3929_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3929_block_003
  · decide
  · rfl

def smallRow3929_block_004 : List Nat := []
def smallPlan3929_block_004 : List RowChoice := [.first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3929_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3929 smallRow3929_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3929_block_004
  · decide
  · rfl

def smallRow3929_small_left : List Nat := smallRow3929_block_000 ++ smallRow3929_block_001
theorem smallRow3929_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3929 smallRow3929_small_left :=
  RowUpperBound.branch smallRow3929_block_000_covers smallRow3929_block_001_covers

def smallRow3929_small_right_right : List Nat := smallRow3929_block_003 ++ smallRow3929_block_004
theorem smallRow3929_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3929 smallRow3929_small_right_right :=
  RowUpperBound.branch smallRow3929_block_003_covers smallRow3929_block_004_covers

def smallRow3929_small_right : List Nat := smallRow3929_block_002 ++ smallRow3929_small_right_right
theorem smallRow3929_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3929 smallRow3929_small_right :=
  RowUpperBound.branch smallRow3929_block_002_covers smallRow3929_small_right_right_covers

def smallRow3929_small : List Nat := smallRow3929_small_left ++ smallRow3929_small_right
theorem smallRow3929_small_covers : RowUpperBound PrimePairVertices.tree_small 3929 smallRow3929_small :=
  RowUpperBound.branch smallRow3929_small_left_covers smallRow3929_small_right_covers

def smallRow3929 : List Nat := smallRow3929_small
theorem smallRow3929_covers : RowUpperBound PrimePairVertices.smallTree 3929 smallRow3929 := smallRow3929_small_covers

def smallRow3931_block_000 : List Nat := []
def smallPlan3931_block_000 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3931_block_000_covers : RowUpperBound PrimePairVertices.tree_block_000 3931 smallRow3931_block_000 := by
  apply row_upper_of_plan _ _ smallPlan3931_block_000
  · decide
  · rfl

def smallRow3931_block_001 : List Nat := []
def smallPlan3931_block_001 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3931_block_001_covers : RowUpperBound PrimePairVertices.tree_block_001 3931 smallRow3931_block_001 := by
  apply row_upper_of_plan _ _ smallPlan3931_block_001
  · decide
  · rfl

def smallRow3931_block_002 : List Nat := [5413, 5449, 5527]
def smallPlan3931_block_002 : List RowChoice := [.first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .second, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .second, .first, .second, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3931_block_002_covers : RowUpperBound PrimePairVertices.tree_block_002 3931 smallRow3931_block_002 := by
  apply row_upper_of_plan _ _ smallPlan3931_block_002
  · decide
  · rfl

def smallRow3931_block_003 : List Nat := [6553, 6661, 7459, 7507, 7561, 7639, 8089]
def smallPlan3931_block_003 : List RowChoice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3931_block_003_covers : RowUpperBound PrimePairVertices.tree_block_003 3931 smallRow3931_block_003 := by
  apply row_upper_of_plan _ _ smallPlan3931_block_003
  · decide
  · rfl

def smallRow3931_block_004 : List Nat := [8233]
def smallPlan3931_block_004 : List RowChoice := [.first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem smallRow3931_block_004_covers : RowUpperBound PrimePairVertices.tree_block_004 3931 smallRow3931_block_004 := by
  apply row_upper_of_plan _ _ smallPlan3931_block_004
  · decide
  · rfl

def smallRow3931_small_left : List Nat := smallRow3931_block_000 ++ smallRow3931_block_001
theorem smallRow3931_small_left_covers : RowUpperBound PrimePairVertices.tree_small_left 3931 smallRow3931_small_left :=
  RowUpperBound.branch smallRow3931_block_000_covers smallRow3931_block_001_covers

def smallRow3931_small_right_right : List Nat := smallRow3931_block_003 ++ smallRow3931_block_004
theorem smallRow3931_small_right_right_covers : RowUpperBound PrimePairVertices.tree_small_right_right 3931 smallRow3931_small_right_right :=
  RowUpperBound.branch smallRow3931_block_003_covers smallRow3931_block_004_covers

def smallRow3931_small_right : List Nat := smallRow3931_block_002 ++ smallRow3931_small_right_right
theorem smallRow3931_small_right_covers : RowUpperBound PrimePairVertices.tree_small_right 3931 smallRow3931_small_right :=
  RowUpperBound.branch smallRow3931_block_002_covers smallRow3931_small_right_right_covers

def smallRow3931_small : List Nat := smallRow3931_small_left ++ smallRow3931_small_right
theorem smallRow3931_small_covers : RowUpperBound PrimePairVertices.tree_small 3931 smallRow3931_small :=
  RowUpperBound.branch smallRow3931_small_left_covers smallRow3931_small_right_covers

def smallRow3931 : List Nat := smallRow3931_small
theorem smallRow3931_covers : RowUpperBound PrimePairVertices.smallTree 3931 smallRow3931 := smallRow3931_small_covers

end ProjectEulerSolutions.PrimePairUpper
