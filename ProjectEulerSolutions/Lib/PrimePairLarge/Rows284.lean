import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9187_small : List Nat := []
theorem row9187_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9187 row9187_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9187_block_005 : List Nat := [9349, 9439, 10357]
def plan9187_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .second, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .second, (.secondFactor 13), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .possible, .second]
set_option maxRecDepth 2048 in
theorem row9187_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9187 row9187_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9187_block_005
  · decide
  · rfl

def row9187_block_006 : List Nat := [11239, 12037, 12211, 12343]
def plan9187_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 31), .second, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .possible, .first, (.secondFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9187_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9187 row9187_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9187_block_006
  · decide
  · rfl

def row9187_block_007 : List Nat := [13177, 14341]
def plan9187_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .possible, (.secondFactor 31), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .second, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 29), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .second, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .second, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .second, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9187_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9187 row9187_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9187_block_007
  · decide
  · rfl

def row9187_block_008 : List Nat := [15643]
def plan9187_block_008 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 19), (.secondFactor 23), .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9187_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9187 row9187_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9187_block_008
  · decide
  · rfl

def row9187_block_009 : List Nat := []
def plan9187_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9187_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9187 row9187_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9187_block_009
  · decide
  · rfl

def row9187_block_010 : List Nat := []
theorem row9187_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9187 row9187_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9187_block_011 : List Nat := []
theorem row9187_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9187 row9187_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9187_block_012 : List Nat := []
theorem row9187_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9187 row9187_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9187_block_013 : List Nat := []
theorem row9187_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9187 row9187_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9187_large_left_left : List Nat := row9187_small ++ row9187_block_005
theorem row9187_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9187 row9187_large_left_left :=
  Budget.RowUpperBound.branch row9187_small_covers row9187_block_005_covers

def row9187_large_left_right_right : List Nat := row9187_block_007 ++ row9187_block_008
theorem row9187_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9187 row9187_large_left_right_right :=
  Budget.RowUpperBound.branch row9187_block_007_covers row9187_block_008_covers

def row9187_large_left_right : List Nat := row9187_block_006 ++ row9187_large_left_right_right
theorem row9187_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9187 row9187_large_left_right :=
  Budget.RowUpperBound.branch row9187_block_006_covers row9187_large_left_right_right_covers

def row9187_large_left : List Nat := row9187_large_left_left ++ row9187_large_left_right
theorem row9187_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9187 row9187_large_left :=
  Budget.RowUpperBound.branch row9187_large_left_left_covers row9187_large_left_right_covers

def row9187_large_right_left : List Nat := row9187_block_009 ++ row9187_block_010
theorem row9187_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9187 row9187_large_right_left :=
  Budget.RowUpperBound.branch row9187_block_009_covers row9187_block_010_covers

def row9187_large_right_right_right : List Nat := row9187_block_012 ++ row9187_block_013
theorem row9187_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9187 row9187_large_right_right_right :=
  Budget.RowUpperBound.branch row9187_block_012_covers row9187_block_013_covers

def row9187_large_right_right : List Nat := row9187_block_011 ++ row9187_large_right_right_right
theorem row9187_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9187 row9187_large_right_right :=
  Budget.RowUpperBound.branch row9187_block_011_covers row9187_large_right_right_right_covers

def row9187_large_right : List Nat := row9187_large_right_left ++ row9187_large_right_right
theorem row9187_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9187 row9187_large_right :=
  Budget.RowUpperBound.branch row9187_large_right_left_covers row9187_large_right_right_covers

def row9187_large : List Nat := row9187_large_left ++ row9187_large_right
theorem row9187_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9187 row9187_large :=
  Budget.RowUpperBound.branch row9187_large_left_covers row9187_large_right_covers

def row9187 : List Nat := row9187_large
theorem row9187_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9187 row9187 := row9187_large_covers

def row9199_small : List Nat := []
theorem row9199_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9199 row9199_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9199_block_005 : List Nat := [9907, 10111, 10333]
def plan9199_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .second, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 31), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, .first, .first, .second, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9199_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9199 row9199_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9199_block_005
  · decide
  · rfl

def row9199_block_006 : List Nat := []
def plan9199_block_006 : List Budget.Choice := [.first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 29), (.secondFactor 31), (.secondFactor 7), .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 29), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9199_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9199 row9199_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9199_block_006
  · decide
  · rfl

def row9199_block_007 : List Nat := [12457, 12589]
def plan9199_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 29), .possible, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 31), .possible, .first, .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 11), .second, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .second, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .second, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row9199_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9199 row9199_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9199_block_007
  · decide
  · rfl

def row9199_block_008 : List Nat := [16267]
def plan9199_block_008 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .second, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 11), (.firstFactor 37), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .second, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9199_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9199 row9199_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9199_block_008
  · decide
  · rfl

def row9199_block_009 : List Nat := [16573]
def plan9199_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .possible, .second, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9199_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9199 row9199_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9199_block_009
  · decide
  · rfl

def row9199_block_010 : List Nat := []
theorem row9199_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9199 row9199_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9199_block_011 : List Nat := []
theorem row9199_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9199 row9199_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9199_block_012 : List Nat := []
theorem row9199_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9199 row9199_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9199_block_013 : List Nat := []
theorem row9199_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9199 row9199_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9199_large_left_left : List Nat := row9199_small ++ row9199_block_005
theorem row9199_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9199 row9199_large_left_left :=
  Budget.RowUpperBound.branch row9199_small_covers row9199_block_005_covers

def row9199_large_left_right_right : List Nat := row9199_block_007 ++ row9199_block_008
theorem row9199_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9199 row9199_large_left_right_right :=
  Budget.RowUpperBound.branch row9199_block_007_covers row9199_block_008_covers

def row9199_large_left_right : List Nat := row9199_block_006 ++ row9199_large_left_right_right
theorem row9199_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9199 row9199_large_left_right :=
  Budget.RowUpperBound.branch row9199_block_006_covers row9199_large_left_right_right_covers

def row9199_large_left : List Nat := row9199_large_left_left ++ row9199_large_left_right
theorem row9199_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9199 row9199_large_left :=
  Budget.RowUpperBound.branch row9199_large_left_left_covers row9199_large_left_right_covers

def row9199_large_right_left : List Nat := row9199_block_009 ++ row9199_block_010
theorem row9199_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9199 row9199_large_right_left :=
  Budget.RowUpperBound.branch row9199_block_009_covers row9199_block_010_covers

def row9199_large_right_right_right : List Nat := row9199_block_012 ++ row9199_block_013
theorem row9199_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9199 row9199_large_right_right_right :=
  Budget.RowUpperBound.branch row9199_block_012_covers row9199_block_013_covers

def row9199_large_right_right : List Nat := row9199_block_011 ++ row9199_large_right_right_right
theorem row9199_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9199 row9199_large_right_right :=
  Budget.RowUpperBound.branch row9199_block_011_covers row9199_large_right_right_right_covers

def row9199_large_right : List Nat := row9199_large_right_left ++ row9199_large_right_right
theorem row9199_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9199 row9199_large_right :=
  Budget.RowUpperBound.branch row9199_large_right_left_covers row9199_large_right_right_covers

def row9199_large : List Nat := row9199_large_left ++ row9199_large_right
theorem row9199_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9199 row9199_large :=
  Budget.RowUpperBound.branch row9199_large_left_covers row9199_large_right_covers

def row9199 : List Nat := row9199_large
theorem row9199_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9199 row9199 := row9199_large_covers

def row9203_small : List Nat := []
theorem row9203_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9203 row9203_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9203_block_005 : List Nat := [9413, 9719]
def plan9203_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), (.firstFactor 37), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), (.secondFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 31), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9203_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9203 row9203_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9203_block_005
  · decide
  · rfl

def row9203_block_006 : List Nat := [10613, 11699, 12251]
def plan9203_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 23), .first, (.secondFactor 23), .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 23), .first, .second, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9203_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9203 row9203_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9203_block_006
  · decide
  · rfl

def row9203_block_007 : List Nat := [12473, 12641]
def plan9203_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .second, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .second, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9203_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9203 row9203_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9203_block_007
  · decide
  · rfl

def row9203_block_008 : List Nat := [14489, 16193]
def plan9203_block_008 : List Budget.Choice := [.first, .possible, .first, .second, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, .second, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 31), .second, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9203_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9203 row9203_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9203_block_008
  · decide
  · rfl

def row9203_block_009 : List Nat := []
def plan9203_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 17), (.secondFactor 11), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9203_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9203 row9203_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9203_block_009
  · decide
  · rfl

def row9203_block_010 : List Nat := []
theorem row9203_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9203 row9203_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9203_block_011 : List Nat := []
theorem row9203_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9203 row9203_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9203_block_012 : List Nat := []
theorem row9203_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9203 row9203_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9203_block_013 : List Nat := []
theorem row9203_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9203 row9203_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9203_large_left_left : List Nat := row9203_small ++ row9203_block_005
theorem row9203_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9203 row9203_large_left_left :=
  Budget.RowUpperBound.branch row9203_small_covers row9203_block_005_covers

def row9203_large_left_right_right : List Nat := row9203_block_007 ++ row9203_block_008
theorem row9203_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9203 row9203_large_left_right_right :=
  Budget.RowUpperBound.branch row9203_block_007_covers row9203_block_008_covers

def row9203_large_left_right : List Nat := row9203_block_006 ++ row9203_large_left_right_right
theorem row9203_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9203 row9203_large_left_right :=
  Budget.RowUpperBound.branch row9203_block_006_covers row9203_large_left_right_right_covers

def row9203_large_left : List Nat := row9203_large_left_left ++ row9203_large_left_right
theorem row9203_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9203 row9203_large_left :=
  Budget.RowUpperBound.branch row9203_large_left_left_covers row9203_large_left_right_covers

def row9203_large_right_left : List Nat := row9203_block_009 ++ row9203_block_010
theorem row9203_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9203 row9203_large_right_left :=
  Budget.RowUpperBound.branch row9203_block_009_covers row9203_block_010_covers

def row9203_large_right_right_right : List Nat := row9203_block_012 ++ row9203_block_013
theorem row9203_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9203 row9203_large_right_right_right :=
  Budget.RowUpperBound.branch row9203_block_012_covers row9203_block_013_covers

def row9203_large_right_right : List Nat := row9203_block_011 ++ row9203_large_right_right_right
theorem row9203_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9203 row9203_large_right_right :=
  Budget.RowUpperBound.branch row9203_block_011_covers row9203_large_right_right_right_covers

def row9203_large_right : List Nat := row9203_large_right_left ++ row9203_large_right_right
theorem row9203_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9203 row9203_large_right :=
  Budget.RowUpperBound.branch row9203_large_right_left_covers row9203_large_right_right_covers

def row9203_large : List Nat := row9203_large_left ++ row9203_large_right
theorem row9203_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9203 row9203_large :=
  Budget.RowUpperBound.branch row9203_large_left_covers row9203_large_right_covers

def row9203 : List Nat := row9203_large
theorem row9203_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9203 row9203 := row9203_large_covers

def row9209_small : List Nat := []
theorem row9209_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9209 row9209_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9209_block_005 : List Nat := [9431, 9629]
def plan9209_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .second, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 31), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 17), .second, (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row9209_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9209 row9209_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9209_block_005
  · decide
  · rfl

def row9209_block_006 : List Nat := [11807, 12107]
def plan9209_block_006 : List Budget.Choice := [(.secondFactor 29), .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 17), .possible, .first, (.firstFactor 11), .first, .second, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9209_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9209 row9209_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9209_block_006
  · decide
  · rfl

def row9209_block_007 : List Nat := [12437, 12743, 14081]
def plan9209_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 19), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .second, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .possible, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9209_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9209 row9209_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9209_block_007
  · decide
  · rfl

def row9209_block_008 : List Nat := [14537, 15887, 16319]
def plan9209_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .second, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .second, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 7), .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .second, (.secondFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .possible, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .possible, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9209_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9209 row9209_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9209_block_008
  · decide
  · rfl

def row9209_block_009 : List Nat := [16649]
def plan9209_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.firstFactor 29), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9209_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9209 row9209_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9209_block_009
  · decide
  · rfl

def row9209_block_010 : List Nat := []
theorem row9209_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9209 row9209_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9209_block_011 : List Nat := []
theorem row9209_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9209 row9209_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9209_block_012 : List Nat := []
theorem row9209_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9209 row9209_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9209_block_013 : List Nat := []
theorem row9209_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9209 row9209_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9209_large_left_left : List Nat := row9209_small ++ row9209_block_005
theorem row9209_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9209 row9209_large_left_left :=
  Budget.RowUpperBound.branch row9209_small_covers row9209_block_005_covers

def row9209_large_left_right_right : List Nat := row9209_block_007 ++ row9209_block_008
theorem row9209_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9209 row9209_large_left_right_right :=
  Budget.RowUpperBound.branch row9209_block_007_covers row9209_block_008_covers

def row9209_large_left_right : List Nat := row9209_block_006 ++ row9209_large_left_right_right
theorem row9209_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9209 row9209_large_left_right :=
  Budget.RowUpperBound.branch row9209_block_006_covers row9209_large_left_right_right_covers

def row9209_large_left : List Nat := row9209_large_left_left ++ row9209_large_left_right
theorem row9209_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9209 row9209_large_left :=
  Budget.RowUpperBound.branch row9209_large_left_left_covers row9209_large_left_right_covers

def row9209_large_right_left : List Nat := row9209_block_009 ++ row9209_block_010
theorem row9209_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9209 row9209_large_right_left :=
  Budget.RowUpperBound.branch row9209_block_009_covers row9209_block_010_covers

def row9209_large_right_right_right : List Nat := row9209_block_012 ++ row9209_block_013
theorem row9209_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9209 row9209_large_right_right_right :=
  Budget.RowUpperBound.branch row9209_block_012_covers row9209_block_013_covers

def row9209_large_right_right : List Nat := row9209_block_011 ++ row9209_large_right_right_right
theorem row9209_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9209 row9209_large_right_right :=
  Budget.RowUpperBound.branch row9209_block_011_covers row9209_large_right_right_right_covers

def row9209_large_right : List Nat := row9209_large_right_left ++ row9209_large_right_right
theorem row9209_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9209 row9209_large_right :=
  Budget.RowUpperBound.branch row9209_large_right_left_covers row9209_large_right_right_covers

def row9209_large : List Nat := row9209_large_left ++ row9209_large_right
theorem row9209_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9209 row9209_large :=
  Budget.RowUpperBound.branch row9209_large_left_covers row9209_large_right_covers

def row9209 : List Nat := row9209_large
theorem row9209_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9209 row9209 := row9209_large_covers

end ProjectEulerSolutions.PrimePairLarge
