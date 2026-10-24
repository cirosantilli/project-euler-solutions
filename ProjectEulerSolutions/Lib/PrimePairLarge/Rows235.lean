import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows069

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7451_small : List Nat := PrimePairUpper.smallRow7451
theorem row7451_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7451 row7451_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7451_covers

def row7451_block_005 : List Nat := [8501, 8747, 8819, 9677, 9833]
def plan7451_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .possible, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, (.secondFactor 37), (.firstFactor 19), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, .first, (.secondFactor 19), (.firstFactor 31), .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .possible, (.firstFactor 23), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7451_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7451 row7451_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7451_block_005
  · decide
  · rfl

def row7451_block_006 : List Nat := [10607, 11273, 11867, 12281]
def plan7451_block_006 : List Budget.Choice := [(.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .possible, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 31), .possible, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row7451_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7451 row7451_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7451_block_006
  · decide
  · rfl

def row7451_block_007 : List Nat := [12611, 12959, 13001]
def plan7451_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .possible, .first, .first, .first, .second, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .second, .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 17), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7451_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7451 row7451_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7451_block_007
  · decide
  · rfl

def row7451_block_008 : List Nat := [14537, 15131, 16139]
def plan7451_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 31), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 11), .first, .possible, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7451_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7451 row7451_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7451_block_008
  · decide
  · rfl

def row7451_block_009 : List Nat := [18251]
def plan7451_block_009 : List Budget.Choice := [.first, (.secondFactor 11), (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row7451_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7451 row7451_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7451_block_009
  · decide
  · rfl

def row7451_block_010 : List Nat := []
def plan7451_block_010 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7451_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7451 row7451_block_010 := by
  apply Budget.row_upper_of_plan _ _ _ plan7451_block_010
  · decide
  · rfl

def row7451_block_011 : List Nat := []
theorem row7451_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7451 row7451_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7451_block_012 : List Nat := []
theorem row7451_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7451 row7451_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7451_block_013 : List Nat := []
theorem row7451_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7451 row7451_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7451_large_left_left : List Nat := row7451_small ++ row7451_block_005
theorem row7451_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7451 row7451_large_left_left :=
  Budget.RowUpperBound.branch row7451_small_covers row7451_block_005_covers

def row7451_large_left_right_right : List Nat := row7451_block_007 ++ row7451_block_008
theorem row7451_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7451 row7451_large_left_right_right :=
  Budget.RowUpperBound.branch row7451_block_007_covers row7451_block_008_covers

def row7451_large_left_right : List Nat := row7451_block_006 ++ row7451_large_left_right_right
theorem row7451_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7451 row7451_large_left_right :=
  Budget.RowUpperBound.branch row7451_block_006_covers row7451_large_left_right_right_covers

def row7451_large_left : List Nat := row7451_large_left_left ++ row7451_large_left_right
theorem row7451_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7451 row7451_large_left :=
  Budget.RowUpperBound.branch row7451_large_left_left_covers row7451_large_left_right_covers

def row7451_large_right_left : List Nat := row7451_block_009 ++ row7451_block_010
theorem row7451_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7451 row7451_large_right_left :=
  Budget.RowUpperBound.branch row7451_block_009_covers row7451_block_010_covers

def row7451_large_right_right_right : List Nat := row7451_block_012 ++ row7451_block_013
theorem row7451_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7451 row7451_large_right_right_right :=
  Budget.RowUpperBound.branch row7451_block_012_covers row7451_block_013_covers

def row7451_large_right_right : List Nat := row7451_block_011 ++ row7451_large_right_right_right
theorem row7451_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7451 row7451_large_right_right :=
  Budget.RowUpperBound.branch row7451_block_011_covers row7451_large_right_right_right_covers

def row7451_large_right : List Nat := row7451_large_right_left ++ row7451_large_right_right
theorem row7451_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7451 row7451_large_right :=
  Budget.RowUpperBound.branch row7451_large_right_left_covers row7451_large_right_right_covers

def row7451_large : List Nat := row7451_large_left ++ row7451_large_right
theorem row7451_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7451 row7451_large :=
  Budget.RowUpperBound.branch row7451_large_left_covers row7451_large_right_covers

def row7451 : List Nat := row7451_large
theorem row7451_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7451 row7451 := row7451_large_covers

def row7457_small : List Nat := PrimePairUpper.smallRow7457
theorem row7457_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7457 row7457_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7457_covers

def row7457_block_005 : List Nat := [8513, 9371, 9497]
def plan7457_block_005 : List Budget.Choice := [.first, .first, .second, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 17), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 13), (.firstFactor 7), .possible, .first, (.firstFactor 31), (.firstFactor 7), .second, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7457_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7457 row7457_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7457_block_005
  · decide
  · rfl

def row7457_block_006 : List Nat := [11261]
def plan7457_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .second, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .second, .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 37), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7457_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7457 row7457_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7457_block_006
  · decide
  · rfl

def row7457_block_007 : List Nat := [13217, 14051, 14327]
def plan7457_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 37), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 23), (.secondFactor 13), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 29), (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7457_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7457 row7457_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7457_block_007
  · decide
  · rfl

def row7457_block_008 : List Nat := [16091]
def plan7457_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7457_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7457 row7457_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7457_block_008
  · decide
  · rfl

def row7457_block_009 : List Nat := [16547, 16889, 17123, 17669, 17939, 17957, 18059]
def plan7457_block_009 : List Budget.Choice := [.first, (.secondFactor 13), .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .possible, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 11), .possible, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 7), .first, .possible, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .possible, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .second, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7457_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7457 row7457_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7457_block_009
  · decide
  · rfl

def row7457_block_010 : List Nat := []
def plan7457_block_010 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7457_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7457 row7457_block_010 := by
  apply Budget.row_upper_of_plan _ _ _ plan7457_block_010
  · decide
  · rfl

def row7457_block_011 : List Nat := []
theorem row7457_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7457 row7457_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7457_block_012 : List Nat := []
theorem row7457_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7457 row7457_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7457_block_013 : List Nat := []
theorem row7457_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7457 row7457_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7457_large_left_left : List Nat := row7457_small ++ row7457_block_005
theorem row7457_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7457 row7457_large_left_left :=
  Budget.RowUpperBound.branch row7457_small_covers row7457_block_005_covers

def row7457_large_left_right_right : List Nat := row7457_block_007 ++ row7457_block_008
theorem row7457_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7457 row7457_large_left_right_right :=
  Budget.RowUpperBound.branch row7457_block_007_covers row7457_block_008_covers

def row7457_large_left_right : List Nat := row7457_block_006 ++ row7457_large_left_right_right
theorem row7457_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7457 row7457_large_left_right :=
  Budget.RowUpperBound.branch row7457_block_006_covers row7457_large_left_right_right_covers

def row7457_large_left : List Nat := row7457_large_left_left ++ row7457_large_left_right
theorem row7457_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7457 row7457_large_left :=
  Budget.RowUpperBound.branch row7457_large_left_left_covers row7457_large_left_right_covers

def row7457_large_right_left : List Nat := row7457_block_009 ++ row7457_block_010
theorem row7457_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7457 row7457_large_right_left :=
  Budget.RowUpperBound.branch row7457_block_009_covers row7457_block_010_covers

def row7457_large_right_right_right : List Nat := row7457_block_012 ++ row7457_block_013
theorem row7457_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7457 row7457_large_right_right_right :=
  Budget.RowUpperBound.branch row7457_block_012_covers row7457_block_013_covers

def row7457_large_right_right : List Nat := row7457_block_011 ++ row7457_large_right_right_right
theorem row7457_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7457 row7457_large_right_right :=
  Budget.RowUpperBound.branch row7457_block_011_covers row7457_large_right_right_right_covers

def row7457_large_right : List Nat := row7457_large_right_left ++ row7457_large_right_right
theorem row7457_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7457 row7457_large_right :=
  Budget.RowUpperBound.branch row7457_large_right_left_covers row7457_large_right_right_covers

def row7457_large : List Nat := row7457_large_left ++ row7457_large_right
theorem row7457_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7457 row7457_large :=
  Budget.RowUpperBound.branch row7457_large_left_covers row7457_large_right_covers

def row7457 : List Nat := row7457_large
theorem row7457_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7457 row7457 := row7457_large_covers

def row7459_small : List Nat := PrimePairUpper.smallRow7459
theorem row7459_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7459 row7459_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7459_covers

def row7459_block_005 : List Nat := [8521, 9319, 9433, 9661, 9901]
def plan7459_block_005 : List Budget.Choice := [(.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .possible, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .second, .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 23), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 7), .first, .possible, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row7459_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7459 row7459_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7459_block_005
  · decide
  · rfl

def row7459_block_006 : List Nat := [10651, 10771, 12073, 12373]
def plan7459_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .possible, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .second, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 31), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row7459_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7459 row7459_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7459_block_006
  · decide
  · rfl

def row7459_block_007 : List Nat := [13009, 13249, 13687, 13711]
def plan7459_block_007 : List Budget.Choice := [(.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 17), .first, .second, (.firstFactor 13), .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, .possible, .first, (.firstFactor 11), .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .second, .possible, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7459_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7459 row7459_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7459_block_007
  · decide
  · rfl

def row7459_block_008 : List Nat := [14731, 16417]
def plan7459_block_008 : List Budget.Choice := [.second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 29), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .second, (.secondFactor 7), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), .possible, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7459_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7459 row7459_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7459_block_008
  · decide
  · rfl

def row7459_block_009 : List Nat := [17377, 17911, 18133, 18379]
def plan7459_block_009 : List Budget.Choice := [(.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 23), (.secondFactor 11), .first, .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row7459_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7459 row7459_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7459_block_009
  · decide
  · rfl

def row7459_block_010 : List Nat := []
def plan7459_block_010 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7459_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7459 row7459_block_010 := by
  apply Budget.row_upper_of_plan _ _ _ plan7459_block_010
  · decide
  · rfl

def row7459_block_011 : List Nat := []
theorem row7459_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7459 row7459_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7459_block_012 : List Nat := []
theorem row7459_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7459 row7459_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7459_block_013 : List Nat := []
theorem row7459_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7459 row7459_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7459_large_left_left : List Nat := row7459_small ++ row7459_block_005
theorem row7459_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7459 row7459_large_left_left :=
  Budget.RowUpperBound.branch row7459_small_covers row7459_block_005_covers

def row7459_large_left_right_right : List Nat := row7459_block_007 ++ row7459_block_008
theorem row7459_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7459 row7459_large_left_right_right :=
  Budget.RowUpperBound.branch row7459_block_007_covers row7459_block_008_covers

def row7459_large_left_right : List Nat := row7459_block_006 ++ row7459_large_left_right_right
theorem row7459_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7459 row7459_large_left_right :=
  Budget.RowUpperBound.branch row7459_block_006_covers row7459_large_left_right_right_covers

def row7459_large_left : List Nat := row7459_large_left_left ++ row7459_large_left_right
theorem row7459_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7459 row7459_large_left :=
  Budget.RowUpperBound.branch row7459_large_left_left_covers row7459_large_left_right_covers

def row7459_large_right_left : List Nat := row7459_block_009 ++ row7459_block_010
theorem row7459_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7459 row7459_large_right_left :=
  Budget.RowUpperBound.branch row7459_block_009_covers row7459_block_010_covers

def row7459_large_right_right_right : List Nat := row7459_block_012 ++ row7459_block_013
theorem row7459_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7459 row7459_large_right_right_right :=
  Budget.RowUpperBound.branch row7459_block_012_covers row7459_block_013_covers

def row7459_large_right_right : List Nat := row7459_block_011 ++ row7459_large_right_right_right
theorem row7459_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7459 row7459_large_right_right :=
  Budget.RowUpperBound.branch row7459_block_011_covers row7459_large_right_right_right_covers

def row7459_large_right : List Nat := row7459_large_right_left ++ row7459_large_right_right
theorem row7459_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7459 row7459_large_right :=
  Budget.RowUpperBound.branch row7459_large_right_left_covers row7459_large_right_right_covers

def row7459_large : List Nat := row7459_large_left ++ row7459_large_right
theorem row7459_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7459 row7459_large :=
  Budget.RowUpperBound.branch row7459_large_left_covers row7459_large_right_covers

def row7459 : List Nat := row7459_large
theorem row7459_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7459 row7459 := row7459_large_covers

def row7477_small : List Nat := PrimePairUpper.smallRow7477
theorem row7477_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7477 row7477_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7477_covers

def row7477_block_005 : List Nat := [9013, 9127, 9391, 9811, 9829, 9907]
def plan7477_block_005 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), .possible, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .second, .second, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .possible, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7477_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7477 row7477_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7477_block_005
  · decide
  · rfl

def row7477_block_006 : List Nat := [10429, 11743, 11941, 12253]
def plan7477_block_006 : List Budget.Choice := [.first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 31), (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 37), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7477_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7477 row7477_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7477_block_006
  · decide
  · rfl

def row7477_block_007 : List Nat := [12799, 13381, 13789, 14197]
def plan7477_block_007 : List Budget.Choice := [.second, (.firstFactor 13), .first, (.firstFactor 23), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 29), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7477_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7477 row7477_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7477_block_007
  · decide
  · rfl

def row7477_block_008 : List Nat := [14533, 14629, 15193, 15217, 15313, 15817, 15973]
def plan7477_block_008 : List Budget.Choice := [(.secondFactor 11), .first, (.secondFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 37), .first, .second, .first, (.secondFactor 7), .possible, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7477_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7477 row7477_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7477_block_008
  · decide
  · rfl

def row7477_block_009 : List Nat := [16603, 16741, 17011, 17293]
def plan7477_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .possible, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 31), .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 29), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 11), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7477_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7477 row7477_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7477_block_009
  · decide
  · rfl

def row7477_block_010 : List Nat := []
theorem row7477_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7477 row7477_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7477_block_011 : List Nat := []
theorem row7477_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7477 row7477_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7477_block_012 : List Nat := []
theorem row7477_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7477 row7477_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7477_block_013 : List Nat := []
theorem row7477_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7477 row7477_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7477_large_left_left : List Nat := row7477_small ++ row7477_block_005
theorem row7477_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7477 row7477_large_left_left :=
  Budget.RowUpperBound.branch row7477_small_covers row7477_block_005_covers

def row7477_large_left_right_right : List Nat := row7477_block_007 ++ row7477_block_008
theorem row7477_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7477 row7477_large_left_right_right :=
  Budget.RowUpperBound.branch row7477_block_007_covers row7477_block_008_covers

def row7477_large_left_right : List Nat := row7477_block_006 ++ row7477_large_left_right_right
theorem row7477_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7477 row7477_large_left_right :=
  Budget.RowUpperBound.branch row7477_block_006_covers row7477_large_left_right_right_covers

def row7477_large_left : List Nat := row7477_large_left_left ++ row7477_large_left_right
theorem row7477_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7477 row7477_large_left :=
  Budget.RowUpperBound.branch row7477_large_left_left_covers row7477_large_left_right_covers

def row7477_large_right_left : List Nat := row7477_block_009 ++ row7477_block_010
theorem row7477_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7477 row7477_large_right_left :=
  Budget.RowUpperBound.branch row7477_block_009_covers row7477_block_010_covers

def row7477_large_right_right_right : List Nat := row7477_block_012 ++ row7477_block_013
theorem row7477_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7477 row7477_large_right_right_right :=
  Budget.RowUpperBound.branch row7477_block_012_covers row7477_block_013_covers

def row7477_large_right_right : List Nat := row7477_block_011 ++ row7477_large_right_right_right
theorem row7477_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7477 row7477_large_right_right :=
  Budget.RowUpperBound.branch row7477_block_011_covers row7477_large_right_right_right_covers

def row7477_large_right : List Nat := row7477_large_right_left ++ row7477_large_right_right
theorem row7477_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7477 row7477_large_right :=
  Budget.RowUpperBound.branch row7477_large_right_left_covers row7477_large_right_right_covers

def row7477_large : List Nat := row7477_large_left ++ row7477_large_right
theorem row7477_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7477 row7477_large :=
  Budget.RowUpperBound.branch row7477_large_left_covers row7477_large_right_covers

def row7477 : List Nat := row7477_large
theorem row7477_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7477 row7477 := row7477_large_covers

end ProjectEulerSolutions.PrimePairLarge
