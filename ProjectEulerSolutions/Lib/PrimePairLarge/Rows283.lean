import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9157_small : List Nat := []
theorem row9157_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9157 row9157_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9157_block_005 : List Nat := [9811, 9907, 10009]
def plan9157_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 13), .first, .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, .second, .second, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9157_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9157 row9157_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9157_block_005
  · decide
  · rfl

def row9157_block_006 : List Nat := [10723, 12211, 12373]
def plan9157_block_006 : List Budget.Choice := [.first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .second, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row9157_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9157 row9157_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9157_block_006
  · decide
  · rfl

def row9157_block_007 : List Nat := [12757, 13399, 13441, 14143]
def plan9157_block_007 : List Budget.Choice := [(.firstFactor 13), .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 7), (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 19), (.firstFactor 19), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9157_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9157 row9157_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9157_block_007
  · decide
  · rfl

def row9157_block_008 : List Nat := [14563, 15679, 15817, 16447]
def plan9157_block_008 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 37), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .possible, .first, (.firstFactor 17), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 23), .second, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9157_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9157 row9157_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9157_block_008
  · decide
  · rfl

def row9157_block_009 : List Nat := []
def plan9157_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 31), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9157_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9157 row9157_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9157_block_009
  · decide
  · rfl

def row9157_block_010 : List Nat := []
theorem row9157_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9157 row9157_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9157_block_011 : List Nat := []
theorem row9157_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9157 row9157_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9157_block_012 : List Nat := []
theorem row9157_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9157 row9157_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9157_block_013 : List Nat := []
theorem row9157_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9157 row9157_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9157_large_left_left : List Nat := row9157_small ++ row9157_block_005
theorem row9157_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9157 row9157_large_left_left :=
  Budget.RowUpperBound.branch row9157_small_covers row9157_block_005_covers

def row9157_large_left_right_right : List Nat := row9157_block_007 ++ row9157_block_008
theorem row9157_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9157 row9157_large_left_right_right :=
  Budget.RowUpperBound.branch row9157_block_007_covers row9157_block_008_covers

def row9157_large_left_right : List Nat := row9157_block_006 ++ row9157_large_left_right_right
theorem row9157_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9157 row9157_large_left_right :=
  Budget.RowUpperBound.branch row9157_block_006_covers row9157_large_left_right_right_covers

def row9157_large_left : List Nat := row9157_large_left_left ++ row9157_large_left_right
theorem row9157_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9157 row9157_large_left :=
  Budget.RowUpperBound.branch row9157_large_left_left_covers row9157_large_left_right_covers

def row9157_large_right_left : List Nat := row9157_block_009 ++ row9157_block_010
theorem row9157_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9157 row9157_large_right_left :=
  Budget.RowUpperBound.branch row9157_block_009_covers row9157_block_010_covers

def row9157_large_right_right_right : List Nat := row9157_block_012 ++ row9157_block_013
theorem row9157_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9157 row9157_large_right_right_right :=
  Budget.RowUpperBound.branch row9157_block_012_covers row9157_block_013_covers

def row9157_large_right_right : List Nat := row9157_block_011 ++ row9157_large_right_right_right
theorem row9157_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9157 row9157_large_right_right :=
  Budget.RowUpperBound.branch row9157_block_011_covers row9157_large_right_right_right_covers

def row9157_large_right : List Nat := row9157_large_right_left ++ row9157_large_right_right
theorem row9157_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9157 row9157_large_right :=
  Budget.RowUpperBound.branch row9157_large_right_left_covers row9157_large_right_right_covers

def row9157_large : List Nat := row9157_large_left ++ row9157_large_right
theorem row9157_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9157 row9157_large :=
  Budget.RowUpperBound.branch row9157_large_left_covers row9157_large_right_covers

def row9157 : List Nat := row9157_large
theorem row9157_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9157 row9157 := row9157_large_covers

def row9161_small : List Nat := []
theorem row9161_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9161 row9161_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9161_block_005 : List Nat := [9533, 10211, 10253, 10271]
def plan9161_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .possible, (.firstFactor 7), .first, (.firstFactor 13), .possible, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row9161_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9161 row9161_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9161_block_005
  · decide
  · rfl

def row9161_block_006 : List Nat := [10487, 10499, 10589, 11117, 11213, 11279, 11633, 11681, 11699, 12101]
def plan9161_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, .possible, .possible, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 17), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 23), .possible, (.firstFactor 17), .first, .possible, .first, .possible, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 19), (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 17), .first, .second, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9161_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9161 row9161_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9161_block_006
  · decide
  · rfl

def row9161_block_007 : List Nat := [12479, 13217, 13469, 13721]
def plan9161_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 31), .first, .first, (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.secondFactor 19), .second, .first, (.firstFactor 7), .first, .second, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row9161_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9161 row9161_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9161_block_007
  · decide
  · rfl

def row9161_block_008 : List Nat := [15107, 16319]
def plan9161_block_008 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .possible, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9161_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9161 row9161_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9161_block_008
  · decide
  · rfl

def row9161_block_009 : List Nat := []
def plan9161_block_009 : List Budget.Choice := [.first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9161_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9161 row9161_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9161_block_009
  · decide
  · rfl

def row9161_block_010 : List Nat := []
theorem row9161_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9161 row9161_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9161_block_011 : List Nat := []
theorem row9161_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9161 row9161_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9161_block_012 : List Nat := []
theorem row9161_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9161 row9161_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9161_block_013 : List Nat := []
theorem row9161_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9161 row9161_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9161_large_left_left : List Nat := row9161_small ++ row9161_block_005
theorem row9161_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9161 row9161_large_left_left :=
  Budget.RowUpperBound.branch row9161_small_covers row9161_block_005_covers

def row9161_large_left_right_right : List Nat := row9161_block_007 ++ row9161_block_008
theorem row9161_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9161 row9161_large_left_right_right :=
  Budget.RowUpperBound.branch row9161_block_007_covers row9161_block_008_covers

def row9161_large_left_right : List Nat := row9161_block_006 ++ row9161_large_left_right_right
theorem row9161_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9161 row9161_large_left_right :=
  Budget.RowUpperBound.branch row9161_block_006_covers row9161_large_left_right_right_covers

def row9161_large_left : List Nat := row9161_large_left_left ++ row9161_large_left_right
theorem row9161_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9161 row9161_large_left :=
  Budget.RowUpperBound.branch row9161_large_left_left_covers row9161_large_left_right_covers

def row9161_large_right_left : List Nat := row9161_block_009 ++ row9161_block_010
theorem row9161_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9161 row9161_large_right_left :=
  Budget.RowUpperBound.branch row9161_block_009_covers row9161_block_010_covers

def row9161_large_right_right_right : List Nat := row9161_block_012 ++ row9161_block_013
theorem row9161_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9161 row9161_large_right_right_right :=
  Budget.RowUpperBound.branch row9161_block_012_covers row9161_block_013_covers

def row9161_large_right_right : List Nat := row9161_block_011 ++ row9161_large_right_right_right
theorem row9161_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9161 row9161_large_right_right :=
  Budget.RowUpperBound.branch row9161_block_011_covers row9161_large_right_right_right_covers

def row9161_large_right : List Nat := row9161_large_right_left ++ row9161_large_right_right
theorem row9161_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9161 row9161_large_right :=
  Budget.RowUpperBound.branch row9161_large_right_left_covers row9161_large_right_right_covers

def row9161_large : List Nat := row9161_large_left ++ row9161_large_right
theorem row9161_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9161 row9161_large :=
  Budget.RowUpperBound.branch row9161_large_left_covers row9161_large_right_covers

def row9161 : List Nat := row9161_large
theorem row9161_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9161 row9161 := row9161_large_covers

def row9173_small : List Nat := []
theorem row9173_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9173 row9173_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9173_block_005 : List Nat := [9677, 9839, 10103]
def plan9173_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 23), (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .second, .first, .first, .first, .first, .second, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .second, (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .second, .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9173_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9173 row9173_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9173_block_005
  · decide
  · rfl

def row9173_block_006 : List Nat := [11657, 11909, 11981]
def plan9173_block_006 : List Budget.Choice := [.second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 19), .second, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row9173_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9173 row9173_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9173_block_006
  · decide
  · rfl

def row9173_block_007 : List Nat := [12899, 13037, 13043, 13499, 13967, 14327]
def plan9173_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .second, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 17), (.firstFactor 37), .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 31), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .possible, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, .first, .possible, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9173_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9173 row9173_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9173_block_007
  · decide
  · rfl

def row9173_block_008 : List Nat := [14753, 16007]
def plan9173_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, (.firstFactor 17), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 31), .first, .first, .second, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .possible, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 29), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9173_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9173 row9173_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9173_block_008
  · decide
  · rfl

def row9173_block_009 : List Nat := [16553]
def plan9173_block_009 : List Budget.Choice := [.first, (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9173_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9173 row9173_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9173_block_009
  · decide
  · rfl

def row9173_block_010 : List Nat := []
theorem row9173_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9173 row9173_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9173_block_011 : List Nat := []
theorem row9173_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9173 row9173_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9173_block_012 : List Nat := []
theorem row9173_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9173 row9173_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9173_block_013 : List Nat := []
theorem row9173_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9173 row9173_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9173_large_left_left : List Nat := row9173_small ++ row9173_block_005
theorem row9173_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9173 row9173_large_left_left :=
  Budget.RowUpperBound.branch row9173_small_covers row9173_block_005_covers

def row9173_large_left_right_right : List Nat := row9173_block_007 ++ row9173_block_008
theorem row9173_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9173 row9173_large_left_right_right :=
  Budget.RowUpperBound.branch row9173_block_007_covers row9173_block_008_covers

def row9173_large_left_right : List Nat := row9173_block_006 ++ row9173_large_left_right_right
theorem row9173_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9173 row9173_large_left_right :=
  Budget.RowUpperBound.branch row9173_block_006_covers row9173_large_left_right_right_covers

def row9173_large_left : List Nat := row9173_large_left_left ++ row9173_large_left_right
theorem row9173_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9173 row9173_large_left :=
  Budget.RowUpperBound.branch row9173_large_left_left_covers row9173_large_left_right_covers

def row9173_large_right_left : List Nat := row9173_block_009 ++ row9173_block_010
theorem row9173_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9173 row9173_large_right_left :=
  Budget.RowUpperBound.branch row9173_block_009_covers row9173_block_010_covers

def row9173_large_right_right_right : List Nat := row9173_block_012 ++ row9173_block_013
theorem row9173_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9173 row9173_large_right_right_right :=
  Budget.RowUpperBound.branch row9173_block_012_covers row9173_block_013_covers

def row9173_large_right_right : List Nat := row9173_block_011 ++ row9173_large_right_right_right
theorem row9173_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9173 row9173_large_right_right :=
  Budget.RowUpperBound.branch row9173_block_011_covers row9173_large_right_right_right_covers

def row9173_large_right : List Nat := row9173_large_right_left ++ row9173_large_right_right
theorem row9173_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9173 row9173_large_right :=
  Budget.RowUpperBound.branch row9173_large_right_left_covers row9173_large_right_right_covers

def row9173_large : List Nat := row9173_large_left ++ row9173_large_right
theorem row9173_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9173 row9173_large :=
  Budget.RowUpperBound.branch row9173_large_left_covers row9173_large_right_covers

def row9173 : List Nat := row9173_large
theorem row9173_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9173 row9173 := row9173_large_covers

def row9181_small : List Nat := []
theorem row9181_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9181 row9181_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9181_block_005 : List Nat := [9769]
def plan9181_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 23), .second, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 17), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 29), .first, .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 7), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9181_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9181 row9181_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9181_block_005
  · decide
  · rfl

def row9181_block_006 : List Nat := [10657, 11743, 11923]
def plan9181_block_006 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .possible, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9181_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9181 row9181_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9181_block_006
  · decide
  · rfl

def row9181_block_007 : List Nat := [12619, 12763, 13177, 14401]
def plan9181_block_007 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .second, .possible, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 29), (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .second, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9181_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9181 row9181_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9181_block_007
  · decide
  · rfl

def row9181_block_008 : List Nat := [15361, 15619, 16453]
def plan9181_block_008 : List Budget.Choice := [.second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .possible, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 29), (.secondFactor 29), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 23), (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 19), .first, .possible, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9181_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9181 row9181_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9181_block_008
  · decide
  · rfl

def row9181_block_009 : List Nat := [16741]
def plan9181_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9181_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9181 row9181_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9181_block_009
  · decide
  · rfl

def row9181_block_010 : List Nat := []
theorem row9181_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9181 row9181_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9181_block_011 : List Nat := []
theorem row9181_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9181 row9181_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9181_block_012 : List Nat := []
theorem row9181_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9181 row9181_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9181_block_013 : List Nat := []
theorem row9181_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9181 row9181_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9181_large_left_left : List Nat := row9181_small ++ row9181_block_005
theorem row9181_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9181 row9181_large_left_left :=
  Budget.RowUpperBound.branch row9181_small_covers row9181_block_005_covers

def row9181_large_left_right_right : List Nat := row9181_block_007 ++ row9181_block_008
theorem row9181_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9181 row9181_large_left_right_right :=
  Budget.RowUpperBound.branch row9181_block_007_covers row9181_block_008_covers

def row9181_large_left_right : List Nat := row9181_block_006 ++ row9181_large_left_right_right
theorem row9181_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9181 row9181_large_left_right :=
  Budget.RowUpperBound.branch row9181_block_006_covers row9181_large_left_right_right_covers

def row9181_large_left : List Nat := row9181_large_left_left ++ row9181_large_left_right
theorem row9181_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9181 row9181_large_left :=
  Budget.RowUpperBound.branch row9181_large_left_left_covers row9181_large_left_right_covers

def row9181_large_right_left : List Nat := row9181_block_009 ++ row9181_block_010
theorem row9181_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9181 row9181_large_right_left :=
  Budget.RowUpperBound.branch row9181_block_009_covers row9181_block_010_covers

def row9181_large_right_right_right : List Nat := row9181_block_012 ++ row9181_block_013
theorem row9181_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9181 row9181_large_right_right_right :=
  Budget.RowUpperBound.branch row9181_block_012_covers row9181_block_013_covers

def row9181_large_right_right : List Nat := row9181_block_011 ++ row9181_large_right_right_right
theorem row9181_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9181 row9181_large_right_right :=
  Budget.RowUpperBound.branch row9181_block_011_covers row9181_large_right_right_right_covers

def row9181_large_right : List Nat := row9181_large_right_left ++ row9181_large_right_right
theorem row9181_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9181 row9181_large_right :=
  Budget.RowUpperBound.branch row9181_large_right_left_covers row9181_large_right_right_covers

def row9181_large : List Nat := row9181_large_left ++ row9181_large_right
theorem row9181_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9181 row9181_large :=
  Budget.RowUpperBound.branch row9181_large_left_covers row9181_large_right_covers

def row9181 : List Nat := row9181_large
theorem row9181_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9181 row9181 := row9181_large_covers

end ProjectEulerSolutions.PrimePairLarge
