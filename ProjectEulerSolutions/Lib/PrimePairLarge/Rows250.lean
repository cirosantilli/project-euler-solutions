import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows077

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7937_small : List Nat := PrimePairUpper.smallRow7937
theorem row7937_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7937 row7937_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7937_covers

def row7937_block_005 : List Nat := [8807, 9413, 10091]
def plan7937_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 19), (.secondFactor 13), .first, .second, .first, (.firstFactor 13), (.secondFactor 37), .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 29), .first, (.secondFactor 23), .possible, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row7937_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7937 row7937_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7937_block_005
  · decide
  · rfl

def row7937_block_006 : List Nat := [10847, 11057, 11777, 12041, 12239]
def plan7937_block_006 : List Budget.Choice := [(.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .second, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 29), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .second, (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7937_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7937 row7937_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7937_block_006
  · decide
  · rfl

def row7937_block_007 : List Nat := [12653, 13103, 13241, 13523, 13859]
def plan7937_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .possible, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .possible, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7937_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7937 row7937_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7937_block_007
  · decide
  · rfl

def row7937_block_008 : List Nat := [15467, 15473, 16073, 16433]
def plan7937_block_008 : List Budget.Choice := [.first, (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 37), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .possible, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 29), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7937_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7937 row7937_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7937_block_008
  · decide
  · rfl

def row7937_block_009 : List Nat := [17387]
def plan7937_block_009 : List Budget.Choice := [.first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7937_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7937 row7937_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7937_block_009
  · decide
  · rfl

def row7937_block_010 : List Nat := []
theorem row7937_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7937 row7937_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7937_block_011 : List Nat := []
theorem row7937_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7937 row7937_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7937_block_012 : List Nat := []
theorem row7937_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7937 row7937_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7937_block_013 : List Nat := []
theorem row7937_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7937 row7937_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7937_large_left_left : List Nat := row7937_small ++ row7937_block_005
theorem row7937_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7937 row7937_large_left_left :=
  Budget.RowUpperBound.branch row7937_small_covers row7937_block_005_covers

def row7937_large_left_right_right : List Nat := row7937_block_007 ++ row7937_block_008
theorem row7937_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7937 row7937_large_left_right_right :=
  Budget.RowUpperBound.branch row7937_block_007_covers row7937_block_008_covers

def row7937_large_left_right : List Nat := row7937_block_006 ++ row7937_large_left_right_right
theorem row7937_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7937 row7937_large_left_right :=
  Budget.RowUpperBound.branch row7937_block_006_covers row7937_large_left_right_right_covers

def row7937_large_left : List Nat := row7937_large_left_left ++ row7937_large_left_right
theorem row7937_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7937 row7937_large_left :=
  Budget.RowUpperBound.branch row7937_large_left_left_covers row7937_large_left_right_covers

def row7937_large_right_left : List Nat := row7937_block_009 ++ row7937_block_010
theorem row7937_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7937 row7937_large_right_left :=
  Budget.RowUpperBound.branch row7937_block_009_covers row7937_block_010_covers

def row7937_large_right_right_right : List Nat := row7937_block_012 ++ row7937_block_013
theorem row7937_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7937 row7937_large_right_right_right :=
  Budget.RowUpperBound.branch row7937_block_012_covers row7937_block_013_covers

def row7937_large_right_right : List Nat := row7937_block_011 ++ row7937_large_right_right_right
theorem row7937_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7937 row7937_large_right_right :=
  Budget.RowUpperBound.branch row7937_block_011_covers row7937_large_right_right_right_covers

def row7937_large_right : List Nat := row7937_large_right_left ++ row7937_large_right_right
theorem row7937_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7937 row7937_large_right :=
  Budget.RowUpperBound.branch row7937_large_right_left_covers row7937_large_right_right_covers

def row7937_large : List Nat := row7937_large_left ++ row7937_large_right
theorem row7937_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7937 row7937_large :=
  Budget.RowUpperBound.branch row7937_large_left_covers row7937_large_right_covers

def row7937 : List Nat := row7937_large
theorem row7937_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7937 row7937 := row7937_large_covers

def row7949_small : List Nat := PrimePairUpper.smallRow7949
theorem row7949_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7949 row7949_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7949_covers

def row7949_block_005 : List Nat := [8501, 9437, 9923, 10253]
def plan7949_block_005 : List Budget.Choice := [.first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .second, (.firstFactor 19), .first, .first, .first, (.secondFactor 37), .first, .first, .possible, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .possible, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7949_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7949 row7949_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7949_block_005
  · decide
  · rfl

def row7949_block_006 : List Nat := [11093, 11597, 11699]
def plan7949_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .possible, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .second, (.secondFactor 19), .second, .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 11), .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row7949_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7949 row7949_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7949_block_006
  · decide
  · rfl

def row7949_block_007 : List Nat := [12689, 13001, 13337]
def plan7949_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 31), (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row7949_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7949 row7949_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7949_block_007
  · decide
  · rfl

def row7949_block_008 : List Nat := [15137, 15383, 15683, 15773, 16061]
def plan7949_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 31), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .possible, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .second, .first, .first, .first, .first, (.firstFactor 37), .second, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), (.firstFactor 37), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7949_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7949 row7949_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7949_block_008
  · decide
  · rfl

def row7949_block_009 : List Nat := [16943]
def plan7949_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 19), (.secondFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7949_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7949 row7949_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7949_block_009
  · decide
  · rfl

def row7949_block_010 : List Nat := []
theorem row7949_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7949 row7949_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7949_block_011 : List Nat := []
theorem row7949_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7949 row7949_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7949_block_012 : List Nat := []
theorem row7949_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7949 row7949_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7949_block_013 : List Nat := []
theorem row7949_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7949 row7949_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7949_large_left_left : List Nat := row7949_small ++ row7949_block_005
theorem row7949_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7949 row7949_large_left_left :=
  Budget.RowUpperBound.branch row7949_small_covers row7949_block_005_covers

def row7949_large_left_right_right : List Nat := row7949_block_007 ++ row7949_block_008
theorem row7949_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7949 row7949_large_left_right_right :=
  Budget.RowUpperBound.branch row7949_block_007_covers row7949_block_008_covers

def row7949_large_left_right : List Nat := row7949_block_006 ++ row7949_large_left_right_right
theorem row7949_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7949 row7949_large_left_right :=
  Budget.RowUpperBound.branch row7949_block_006_covers row7949_large_left_right_right_covers

def row7949_large_left : List Nat := row7949_large_left_left ++ row7949_large_left_right
theorem row7949_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7949 row7949_large_left :=
  Budget.RowUpperBound.branch row7949_large_left_left_covers row7949_large_left_right_covers

def row7949_large_right_left : List Nat := row7949_block_009 ++ row7949_block_010
theorem row7949_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7949 row7949_large_right_left :=
  Budget.RowUpperBound.branch row7949_block_009_covers row7949_block_010_covers

def row7949_large_right_right_right : List Nat := row7949_block_012 ++ row7949_block_013
theorem row7949_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7949 row7949_large_right_right_right :=
  Budget.RowUpperBound.branch row7949_block_012_covers row7949_block_013_covers

def row7949_large_right_right : List Nat := row7949_block_011 ++ row7949_large_right_right_right
theorem row7949_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7949 row7949_large_right_right :=
  Budget.RowUpperBound.branch row7949_block_011_covers row7949_large_right_right_right_covers

def row7949_large_right : List Nat := row7949_large_right_left ++ row7949_large_right_right
theorem row7949_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7949 row7949_large_right :=
  Budget.RowUpperBound.branch row7949_large_right_left_covers row7949_large_right_right_covers

def row7949_large : List Nat := row7949_large_left ++ row7949_large_right
theorem row7949_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7949 row7949_large :=
  Budget.RowUpperBound.branch row7949_large_left_covers row7949_large_right_covers

def row7949 : List Nat := row7949_large
theorem row7949_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7949 row7949 := row7949_large_covers

def row7951_small : List Nat := PrimePairUpper.smallRow7951
theorem row7951_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7951 row7951_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7951_covers

def row7951_block_005 : List Nat := [8623, 8641, 8779, 9001, 9439, 9463]
def plan7951_block_005 : List Budget.Choice := [(.secondFactor 23), .second, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 19), .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 29), .first, (.secondFactor 7), .second, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .possible, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7951_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7951 row7951_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7951_block_005
  · decide
  · rfl

def row7951_block_006 : List Nat := [11113, 11257]
def plan7951_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .second, .first, .second, .first, .second, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), .possible, .first, .first, .first, (.secondFactor 23), (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 13), (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 37), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7951_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7951 row7951_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7951_block_006
  · decide
  · rfl

def row7951_block_007 : List Nat := [12907, 14197]
def plan7951_block_007 : List Budget.Choice := [(.secondFactor 11), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .second, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 17), .second, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 29), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, .second, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 7), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .second, (.secondFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row7951_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7951 row7951_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7951_block_007
  · decide
  · rfl

def row7951_block_008 : List Nat := [14713, 15679, 15889, 16111, 16363, 16411]
def plan7951_block_008 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .second, .first, (.secondFactor 17), (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 19), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 17), .second, .first, .first, .first, .possible, .second, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .possible, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 11), .possible, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7951_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7951 row7951_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7951_block_008
  · decide
  · rfl

def row7951_block_009 : List Nat := [16573, 17293, 17599, 17989]
def plan7951_block_009 : List Budget.Choice := [(.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .possible, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 17), (.secondFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 31), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .second, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 31), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7951_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7951 row7951_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7951_block_009
  · decide
  · rfl

def row7951_block_010 : List Nat := []
theorem row7951_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7951 row7951_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7951_block_011 : List Nat := []
theorem row7951_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7951 row7951_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7951_block_012 : List Nat := []
theorem row7951_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7951 row7951_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7951_block_013 : List Nat := []
theorem row7951_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7951 row7951_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7951_large_left_left : List Nat := row7951_small ++ row7951_block_005
theorem row7951_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7951 row7951_large_left_left :=
  Budget.RowUpperBound.branch row7951_small_covers row7951_block_005_covers

def row7951_large_left_right_right : List Nat := row7951_block_007 ++ row7951_block_008
theorem row7951_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7951 row7951_large_left_right_right :=
  Budget.RowUpperBound.branch row7951_block_007_covers row7951_block_008_covers

def row7951_large_left_right : List Nat := row7951_block_006 ++ row7951_large_left_right_right
theorem row7951_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7951 row7951_large_left_right :=
  Budget.RowUpperBound.branch row7951_block_006_covers row7951_large_left_right_right_covers

def row7951_large_left : List Nat := row7951_large_left_left ++ row7951_large_left_right
theorem row7951_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7951 row7951_large_left :=
  Budget.RowUpperBound.branch row7951_large_left_left_covers row7951_large_left_right_covers

def row7951_large_right_left : List Nat := row7951_block_009 ++ row7951_block_010
theorem row7951_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7951 row7951_large_right_left :=
  Budget.RowUpperBound.branch row7951_block_009_covers row7951_block_010_covers

def row7951_large_right_right_right : List Nat := row7951_block_012 ++ row7951_block_013
theorem row7951_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7951 row7951_large_right_right_right :=
  Budget.RowUpperBound.branch row7951_block_012_covers row7951_block_013_covers

def row7951_large_right_right : List Nat := row7951_block_011 ++ row7951_large_right_right_right
theorem row7951_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7951 row7951_large_right_right :=
  Budget.RowUpperBound.branch row7951_block_011_covers row7951_large_right_right_right_covers

def row7951_large_right : List Nat := row7951_large_right_left ++ row7951_large_right_right
theorem row7951_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7951 row7951_large_right :=
  Budget.RowUpperBound.branch row7951_large_right_left_covers row7951_large_right_right_covers

def row7951_large : List Nat := row7951_large_left ++ row7951_large_right
theorem row7951_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7951 row7951_large :=
  Budget.RowUpperBound.branch row7951_large_left_covers row7951_large_right_covers

def row7951 : List Nat := row7951_large
theorem row7951_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7951 row7951 := row7951_large_covers

def row7963_small : List Nat := PrimePairUpper.smallRow7963
theorem row7963_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7963 row7963_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7963_covers

def row7963_block_005 : List Nat := [9181, 9187, 9199, 10111]
def plan7963_block_005 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, .possible, .possible, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7963_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7963 row7963_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7963_block_005
  · decide
  · rfl

def row7963_block_006 : List Nat := [10453, 11239, 11443, 11587]
def plan7963_block_006 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row7963_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7963 row7963_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7963_block_006
  · decide
  · rfl

def row7963_block_007 : List Nat := [12457, 12967, 14011]
def plan7963_block_007 : List Budget.Choice := [(.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .possible, (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .second, .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7963_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7963 row7963_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7963_block_007
  · decide
  · rfl

def row7963_block_008 : List Nat := [14551, 14593, 15391, 15439, 15601]
def plan7963_block_008 : List Budget.Choice := [.second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), .second, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, (.secondFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 23), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, .first, .second, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7963_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7963 row7963_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7963_block_008
  · decide
  · rfl

def row7963_block_009 : List Nat := [17377, 17881]
def plan7963_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .second, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .possible, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, (.firstFactor 29), .second, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7963_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7963 row7963_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7963_block_009
  · decide
  · rfl

def row7963_block_010 : List Nat := []
theorem row7963_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7963 row7963_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7963_block_011 : List Nat := []
theorem row7963_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7963 row7963_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7963_block_012 : List Nat := []
theorem row7963_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7963 row7963_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7963_block_013 : List Nat := []
theorem row7963_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7963 row7963_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7963_large_left_left : List Nat := row7963_small ++ row7963_block_005
theorem row7963_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7963 row7963_large_left_left :=
  Budget.RowUpperBound.branch row7963_small_covers row7963_block_005_covers

def row7963_large_left_right_right : List Nat := row7963_block_007 ++ row7963_block_008
theorem row7963_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7963 row7963_large_left_right_right :=
  Budget.RowUpperBound.branch row7963_block_007_covers row7963_block_008_covers

def row7963_large_left_right : List Nat := row7963_block_006 ++ row7963_large_left_right_right
theorem row7963_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7963 row7963_large_left_right :=
  Budget.RowUpperBound.branch row7963_block_006_covers row7963_large_left_right_right_covers

def row7963_large_left : List Nat := row7963_large_left_left ++ row7963_large_left_right
theorem row7963_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7963 row7963_large_left :=
  Budget.RowUpperBound.branch row7963_large_left_left_covers row7963_large_left_right_covers

def row7963_large_right_left : List Nat := row7963_block_009 ++ row7963_block_010
theorem row7963_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7963 row7963_large_right_left :=
  Budget.RowUpperBound.branch row7963_block_009_covers row7963_block_010_covers

def row7963_large_right_right_right : List Nat := row7963_block_012 ++ row7963_block_013
theorem row7963_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7963 row7963_large_right_right_right :=
  Budget.RowUpperBound.branch row7963_block_012_covers row7963_block_013_covers

def row7963_large_right_right : List Nat := row7963_block_011 ++ row7963_large_right_right_right
theorem row7963_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7963 row7963_large_right_right :=
  Budget.RowUpperBound.branch row7963_block_011_covers row7963_large_right_right_right_covers

def row7963_large_right : List Nat := row7963_large_right_left ++ row7963_large_right_right
theorem row7963_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7963 row7963_large_right :=
  Budget.RowUpperBound.branch row7963_large_right_left_covers row7963_large_right_right_covers

def row7963_large : List Nat := row7963_large_left ++ row7963_large_right
theorem row7963_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7963 row7963_large :=
  Budget.RowUpperBound.branch row7963_large_left_covers row7963_large_right_covers

def row7963 : List Nat := row7963_large
theorem row7963_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7963 row7963 := row7963_large_covers

end ProjectEulerSolutions.PrimePairLarge
