import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8887_small : List Nat := []
theorem row8887_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8887 row8887_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8887_block_005 : List Nat := [9601]
def plan8887_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 23), (.secondFactor 29), .first, .first, (.secondFactor 7), .second, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8887_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8887 row8887_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8887_block_005
  · decide
  · rfl

def row8887_block_006 : List Nat := [12073, 12301]
def plan8887_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .possible, .first, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8887_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8887 row8887_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8887_block_006
  · decide
  · rfl

def row8887_block_007 : List Nat := [12637, 13093, 13183, 13219, 14149]
def plan8887_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .possible, .first, .first, .possible, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .second, .second, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), (.secondFactor 29), .possible, .first, .first, .second, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8887_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8887 row8887_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8887_block_007
  · decide
  · rfl

def row8887_block_008 : List Nat := [15031, 15277, 15859]
def plan8887_block_008 : List Budget.Choice := [(.firstFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .possible, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), (.firstFactor 23), (.secondFactor 17), (.firstFactor 13), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8887_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8887 row8887_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8887_block_008
  · decide
  · rfl

def row8887_block_009 : List Nat := []
def plan8887_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.secondFactor 19), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8887_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8887 row8887_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8887_block_009
  · decide
  · rfl

def row8887_block_010 : List Nat := []
theorem row8887_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8887 row8887_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8887_block_011 : List Nat := []
theorem row8887_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8887 row8887_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8887_block_012 : List Nat := []
theorem row8887_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8887 row8887_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8887_block_013 : List Nat := []
theorem row8887_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8887 row8887_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8887_large_left_left : List Nat := row8887_small ++ row8887_block_005
theorem row8887_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8887 row8887_large_left_left :=
  Budget.RowUpperBound.branch row8887_small_covers row8887_block_005_covers

def row8887_large_left_right_right : List Nat := row8887_block_007 ++ row8887_block_008
theorem row8887_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8887 row8887_large_left_right_right :=
  Budget.RowUpperBound.branch row8887_block_007_covers row8887_block_008_covers

def row8887_large_left_right : List Nat := row8887_block_006 ++ row8887_large_left_right_right
theorem row8887_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8887 row8887_large_left_right :=
  Budget.RowUpperBound.branch row8887_block_006_covers row8887_large_left_right_right_covers

def row8887_large_left : List Nat := row8887_large_left_left ++ row8887_large_left_right
theorem row8887_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8887 row8887_large_left :=
  Budget.RowUpperBound.branch row8887_large_left_left_covers row8887_large_left_right_covers

def row8887_large_right_left : List Nat := row8887_block_009 ++ row8887_block_010
theorem row8887_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8887 row8887_large_right_left :=
  Budget.RowUpperBound.branch row8887_block_009_covers row8887_block_010_covers

def row8887_large_right_right_right : List Nat := row8887_block_012 ++ row8887_block_013
theorem row8887_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8887 row8887_large_right_right_right :=
  Budget.RowUpperBound.branch row8887_block_012_covers row8887_block_013_covers

def row8887_large_right_right : List Nat := row8887_block_011 ++ row8887_large_right_right_right
theorem row8887_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8887 row8887_large_right_right :=
  Budget.RowUpperBound.branch row8887_block_011_covers row8887_large_right_right_right_covers

def row8887_large_right : List Nat := row8887_large_right_left ++ row8887_large_right_right
theorem row8887_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8887 row8887_large_right :=
  Budget.RowUpperBound.branch row8887_large_right_left_covers row8887_large_right_right_covers

def row8887_large : List Nat := row8887_large_left ++ row8887_large_right
theorem row8887_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8887 row8887_large :=
  Budget.RowUpperBound.branch row8887_large_left_covers row8887_large_right_covers

def row8887 : List Nat := row8887_large
theorem row8887_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8887 row8887 := row8887_large_covers

def row8893_small : List Nat := []
theorem row8893_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8893 row8893_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8893_block_005 : List Nat := [9187, 9421, 9649, 9973]
def plan8893_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .possible, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .second]
set_option maxRecDepth 2048 in
theorem row8893_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8893 row8893_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8893_block_005
  · decide
  · rfl

def row8893_block_006 : List Nat := [11953, 12037, 12301]
def plan8893_block_006 : List Budget.Choice := [.first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 19), .first, .second, (.secondFactor 7), .first, .first, .first, .second, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8893_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8893 row8893_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8893_block_006
  · decide
  · rfl

def row8893_block_007 : List Nat := [12577, 13627, 14011]
def plan8893_block_007 : List Budget.Choice := [.first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .second, .first, .possible, .second, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .possible, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8893_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8893 row8893_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8893_block_007
  · decide
  · rfl

def row8893_block_008 : List Nat := [14731, 15727, 16063, 16069]
def plan8893_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .second, (.secondFactor 31), .second, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .possible, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 13), .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8893_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8893 row8893_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8893_block_008
  · decide
  · rfl

def row8893_block_009 : List Nat := [17041]
def plan8893_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .second, (.firstFactor 11), .first, .first, .first, .first, .possible, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8893_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8893 row8893_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8893_block_009
  · decide
  · rfl

def row8893_block_010 : List Nat := []
theorem row8893_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8893 row8893_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8893_block_011 : List Nat := []
theorem row8893_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8893 row8893_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8893_block_012 : List Nat := []
theorem row8893_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8893 row8893_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8893_block_013 : List Nat := []
theorem row8893_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8893 row8893_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8893_large_left_left : List Nat := row8893_small ++ row8893_block_005
theorem row8893_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8893 row8893_large_left_left :=
  Budget.RowUpperBound.branch row8893_small_covers row8893_block_005_covers

def row8893_large_left_right_right : List Nat := row8893_block_007 ++ row8893_block_008
theorem row8893_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8893 row8893_large_left_right_right :=
  Budget.RowUpperBound.branch row8893_block_007_covers row8893_block_008_covers

def row8893_large_left_right : List Nat := row8893_block_006 ++ row8893_large_left_right_right
theorem row8893_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8893 row8893_large_left_right :=
  Budget.RowUpperBound.branch row8893_block_006_covers row8893_large_left_right_right_covers

def row8893_large_left : List Nat := row8893_large_left_left ++ row8893_large_left_right
theorem row8893_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8893 row8893_large_left :=
  Budget.RowUpperBound.branch row8893_large_left_left_covers row8893_large_left_right_covers

def row8893_large_right_left : List Nat := row8893_block_009 ++ row8893_block_010
theorem row8893_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8893 row8893_large_right_left :=
  Budget.RowUpperBound.branch row8893_block_009_covers row8893_block_010_covers

def row8893_large_right_right_right : List Nat := row8893_block_012 ++ row8893_block_013
theorem row8893_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8893 row8893_large_right_right_right :=
  Budget.RowUpperBound.branch row8893_block_012_covers row8893_block_013_covers

def row8893_large_right_right : List Nat := row8893_block_011 ++ row8893_large_right_right_right
theorem row8893_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8893 row8893_large_right_right :=
  Budget.RowUpperBound.branch row8893_block_011_covers row8893_large_right_right_right_covers

def row8893_large_right : List Nat := row8893_large_right_left ++ row8893_large_right_right
theorem row8893_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8893 row8893_large_right :=
  Budget.RowUpperBound.branch row8893_large_right_left_covers row8893_large_right_right_covers

def row8893_large : List Nat := row8893_large_left ++ row8893_large_right
theorem row8893_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8893 row8893_large :=
  Budget.RowUpperBound.branch row8893_large_left_covers row8893_large_right_covers

def row8893 : List Nat := row8893_large
theorem row8893_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8893 row8893 := row8893_large_covers

def row8923_small : List Nat := []
theorem row8923_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8923 row8923_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8923_block_005 : List Nat := [9043, 9091, 9343, 9721, 9883, 10099]
def plan8923_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 31), .first, (.firstFactor 19), .possible, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .possible, .first, .first, .second, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 37), (.secondFactor 19), .first, (.secondFactor 29), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row8923_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8923 row8923_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8923_block_005
  · decide
  · rfl

def row8923_block_006 : List Nat := [10711, 11593]
def plan8923_block_006 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, .first, .possible, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8923_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8923 row8923_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8923_block_006
  · decide
  · rfl

def row8923_block_007 : List Nat := [12601, 13183, 14083]
def plan8923_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .possible, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .second, (.secondFactor 7), .second, .first, .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 31), (.firstFactor 7), .second, .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row8923_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8923 row8923_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8923_block_007
  · decide
  · rfl

def row8923_block_008 : List Nat := [15973]
def plan8923_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .second, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 29), .first, .first, (.secondFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8923_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8923 row8923_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8923_block_008
  · decide
  · rfl

def row8923_block_009 : List Nat := []
def plan8923_block_009 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, .second, (.secondFactor 17), (.firstFactor 29), .second, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8923_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8923 row8923_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8923_block_009
  · decide
  · rfl

def row8923_block_010 : List Nat := []
theorem row8923_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8923 row8923_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8923_block_011 : List Nat := []
theorem row8923_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8923 row8923_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8923_block_012 : List Nat := []
theorem row8923_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8923 row8923_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8923_block_013 : List Nat := []
theorem row8923_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8923 row8923_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8923_large_left_left : List Nat := row8923_small ++ row8923_block_005
theorem row8923_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8923 row8923_large_left_left :=
  Budget.RowUpperBound.branch row8923_small_covers row8923_block_005_covers

def row8923_large_left_right_right : List Nat := row8923_block_007 ++ row8923_block_008
theorem row8923_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8923 row8923_large_left_right_right :=
  Budget.RowUpperBound.branch row8923_block_007_covers row8923_block_008_covers

def row8923_large_left_right : List Nat := row8923_block_006 ++ row8923_large_left_right_right
theorem row8923_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8923 row8923_large_left_right :=
  Budget.RowUpperBound.branch row8923_block_006_covers row8923_large_left_right_right_covers

def row8923_large_left : List Nat := row8923_large_left_left ++ row8923_large_left_right
theorem row8923_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8923 row8923_large_left :=
  Budget.RowUpperBound.branch row8923_large_left_left_covers row8923_large_left_right_covers

def row8923_large_right_left : List Nat := row8923_block_009 ++ row8923_block_010
theorem row8923_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8923 row8923_large_right_left :=
  Budget.RowUpperBound.branch row8923_block_009_covers row8923_block_010_covers

def row8923_large_right_right_right : List Nat := row8923_block_012 ++ row8923_block_013
theorem row8923_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8923 row8923_large_right_right_right :=
  Budget.RowUpperBound.branch row8923_block_012_covers row8923_block_013_covers

def row8923_large_right_right : List Nat := row8923_block_011 ++ row8923_large_right_right_right
theorem row8923_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8923 row8923_large_right_right :=
  Budget.RowUpperBound.branch row8923_block_011_covers row8923_large_right_right_right_covers

def row8923_large_right : List Nat := row8923_large_right_left ++ row8923_large_right_right
theorem row8923_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8923 row8923_large_right :=
  Budget.RowUpperBound.branch row8923_large_right_left_covers row8923_large_right_right_covers

def row8923_large : List Nat := row8923_large_left ++ row8923_large_right
theorem row8923_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8923 row8923_large :=
  Budget.RowUpperBound.branch row8923_large_left_covers row8923_large_right_covers

def row8923 : List Nat := row8923_large
theorem row8923_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8923 row8923 := row8923_large_covers

def row8929_small : List Nat := []
theorem row8929_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8929 row8929_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8929_block_005 : List Nat := [9319, 9733, 9817, 10069, 10141]
def plan8929_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .possible, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8929_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8929 row8929_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8929_block_005
  · decide
  · rfl

def row8929_block_006 : List Nat := [10861, 10987, 11527, 11863]
def plan8929_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 11), .second, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 23), (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .second, .first, .second, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row8929_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8929 row8929_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8929_block_006
  · decide
  · rfl

def row8929_block_007 : List Nat := [12577]
def plan8929_block_007 : List Budget.Choice := [(.secondFactor 23), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 37), (.firstFactor 23), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8929_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8929 row8929_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8929_block_007
  · decide
  · rfl

def row8929_block_008 : List Nat := [14887, 14983, 15031, 16417]
def plan8929_block_008 : List Budget.Choice := [(.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .possible, (.firstFactor 19), .first, .possible, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .second, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 13), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8929_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8929 row8929_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8929_block_008
  · decide
  · rfl

def row8929_block_009 : List Nat := [16903]
def plan8929_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8929_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8929 row8929_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8929_block_009
  · decide
  · rfl

def row8929_block_010 : List Nat := []
theorem row8929_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8929 row8929_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8929_block_011 : List Nat := []
theorem row8929_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8929 row8929_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8929_block_012 : List Nat := []
theorem row8929_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8929 row8929_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8929_block_013 : List Nat := []
theorem row8929_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8929 row8929_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8929_large_left_left : List Nat := row8929_small ++ row8929_block_005
theorem row8929_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8929 row8929_large_left_left :=
  Budget.RowUpperBound.branch row8929_small_covers row8929_block_005_covers

def row8929_large_left_right_right : List Nat := row8929_block_007 ++ row8929_block_008
theorem row8929_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8929 row8929_large_left_right_right :=
  Budget.RowUpperBound.branch row8929_block_007_covers row8929_block_008_covers

def row8929_large_left_right : List Nat := row8929_block_006 ++ row8929_large_left_right_right
theorem row8929_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8929 row8929_large_left_right :=
  Budget.RowUpperBound.branch row8929_block_006_covers row8929_large_left_right_right_covers

def row8929_large_left : List Nat := row8929_large_left_left ++ row8929_large_left_right
theorem row8929_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8929 row8929_large_left :=
  Budget.RowUpperBound.branch row8929_large_left_left_covers row8929_large_left_right_covers

def row8929_large_right_left : List Nat := row8929_block_009 ++ row8929_block_010
theorem row8929_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8929 row8929_large_right_left :=
  Budget.RowUpperBound.branch row8929_block_009_covers row8929_block_010_covers

def row8929_large_right_right_right : List Nat := row8929_block_012 ++ row8929_block_013
theorem row8929_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8929 row8929_large_right_right_right :=
  Budget.RowUpperBound.branch row8929_block_012_covers row8929_block_013_covers

def row8929_large_right_right : List Nat := row8929_block_011 ++ row8929_large_right_right_right
theorem row8929_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8929 row8929_large_right_right :=
  Budget.RowUpperBound.branch row8929_block_011_covers row8929_large_right_right_right_covers

def row8929_large_right : List Nat := row8929_large_right_left ++ row8929_large_right_right
theorem row8929_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8929 row8929_large_right :=
  Budget.RowUpperBound.branch row8929_large_right_left_covers row8929_large_right_right_covers

def row8929_large : List Nat := row8929_large_left ++ row8929_large_right
theorem row8929_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8929 row8929_large :=
  Budget.RowUpperBound.branch row8929_large_left_covers row8929_large_right_covers

def row8929 : List Nat := row8929_large
theorem row8929_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8929 row8929 := row8929_large_covers

end ProjectEulerSolutions.PrimePairLarge
