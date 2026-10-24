import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9337_small : List Nat := []
theorem row9337_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9337 row9337_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9337_block_005 : List Nat := []
def plan9337_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, .second, (.secondFactor 19), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 29), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9337_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9337 row9337_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9337_block_005
  · decide
  · rfl

def row9337_block_006 : List Nat := [10909, 10993, 11719, 11953]
def plan9337_block_006 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .second, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 31), (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .possible, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 29), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9337_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9337 row9337_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9337_block_006
  · decide
  · rfl

def row9337_block_007 : List Nat := [13171]
def plan9337_block_007 : List Budget.Choice := [(.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .second, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 23), (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .second, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .second]
set_option maxRecDepth 2048 in
theorem row9337_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9337 row9337_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9337_block_007
  · decide
  · rfl

def row9337_block_008 : List Nat := [14563, 15439]
def plan9337_block_008 : List Budget.Choice := [(.firstFactor 29), .first, .second, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .second, (.secondFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9337_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9337 row9337_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9337_block_008
  · decide
  · rfl

def row9337_block_009 : List Nat := []
def plan9337_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9337_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9337 row9337_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9337_block_009
  · decide
  · rfl

def row9337_block_010 : List Nat := []
theorem row9337_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9337 row9337_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9337_block_011 : List Nat := []
theorem row9337_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9337 row9337_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9337_block_012 : List Nat := []
theorem row9337_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9337 row9337_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9337_block_013 : List Nat := []
theorem row9337_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9337 row9337_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9337_large_left_left : List Nat := row9337_small ++ row9337_block_005
theorem row9337_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9337 row9337_large_left_left :=
  Budget.RowUpperBound.branch row9337_small_covers row9337_block_005_covers

def row9337_large_left_right_right : List Nat := row9337_block_007 ++ row9337_block_008
theorem row9337_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9337 row9337_large_left_right_right :=
  Budget.RowUpperBound.branch row9337_block_007_covers row9337_block_008_covers

def row9337_large_left_right : List Nat := row9337_block_006 ++ row9337_large_left_right_right
theorem row9337_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9337 row9337_large_left_right :=
  Budget.RowUpperBound.branch row9337_block_006_covers row9337_large_left_right_right_covers

def row9337_large_left : List Nat := row9337_large_left_left ++ row9337_large_left_right
theorem row9337_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9337 row9337_large_left :=
  Budget.RowUpperBound.branch row9337_large_left_left_covers row9337_large_left_right_covers

def row9337_large_right_left : List Nat := row9337_block_009 ++ row9337_block_010
theorem row9337_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9337 row9337_large_right_left :=
  Budget.RowUpperBound.branch row9337_block_009_covers row9337_block_010_covers

def row9337_large_right_right_right : List Nat := row9337_block_012 ++ row9337_block_013
theorem row9337_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9337 row9337_large_right_right_right :=
  Budget.RowUpperBound.branch row9337_block_012_covers row9337_block_013_covers

def row9337_large_right_right : List Nat := row9337_block_011 ++ row9337_large_right_right_right
theorem row9337_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9337 row9337_large_right_right :=
  Budget.RowUpperBound.branch row9337_block_011_covers row9337_large_right_right_right_covers

def row9337_large_right : List Nat := row9337_large_right_left ++ row9337_large_right_right
theorem row9337_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9337 row9337_large_right :=
  Budget.RowUpperBound.branch row9337_large_right_left_covers row9337_large_right_right_covers

def row9337_large : List Nat := row9337_large_left ++ row9337_large_right
theorem row9337_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9337 row9337_large :=
  Budget.RowUpperBound.branch row9337_large_left_covers row9337_large_right_covers

def row9337 : List Nat := row9337_large
theorem row9337_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9337 row9337 := row9337_large_covers

def row9341_small : List Nat := []
theorem row9341_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9341 row9341_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9341_block_005 : List Nat := [9497, 10061, 10211]
def plan9341_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .possible, .first, (.firstFactor 13), .second, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .second, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), (.secondFactor 7), .possible, .second, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9341_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9341 row9341_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9341_block_005
  · decide
  · rfl

def row9341_block_006 : List Nat := [10427, 11027, 11279, 11681]
def plan9341_block_006 : List Budget.Choice := [(.secondFactor 19), .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 31), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 19), .second, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9341_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9341 row9341_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9341_block_006
  · decide
  · rfl

def row9341_block_007 : List Nat := [12437, 12497, 13043, 13163, 14033]
def plan9341_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .possible, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .second, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 13), .second, .first, (.firstFactor 17), (.firstFactor 19), .second, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .possible, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), (.secondFactor 13), .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .possible, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9341_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9341 row9341_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9341_block_007
  · decide
  · rfl

def row9341_block_008 : List Nat := [14747, 14891, 15131, 15971, 16061, 16427]
def plan9341_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 37), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .second, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 19), .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 11), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9341_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9341 row9341_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9341_block_008
  · decide
  · rfl

def row9341_block_009 : List Nat := []
def plan9341_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9341_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9341 row9341_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9341_block_009
  · decide
  · rfl

def row9341_block_010 : List Nat := []
theorem row9341_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9341 row9341_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9341_block_011 : List Nat := []
theorem row9341_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9341 row9341_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9341_block_012 : List Nat := []
theorem row9341_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9341 row9341_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9341_block_013 : List Nat := []
theorem row9341_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9341 row9341_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9341_large_left_left : List Nat := row9341_small ++ row9341_block_005
theorem row9341_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9341 row9341_large_left_left :=
  Budget.RowUpperBound.branch row9341_small_covers row9341_block_005_covers

def row9341_large_left_right_right : List Nat := row9341_block_007 ++ row9341_block_008
theorem row9341_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9341 row9341_large_left_right_right :=
  Budget.RowUpperBound.branch row9341_block_007_covers row9341_block_008_covers

def row9341_large_left_right : List Nat := row9341_block_006 ++ row9341_large_left_right_right
theorem row9341_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9341 row9341_large_left_right :=
  Budget.RowUpperBound.branch row9341_block_006_covers row9341_large_left_right_right_covers

def row9341_large_left : List Nat := row9341_large_left_left ++ row9341_large_left_right
theorem row9341_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9341 row9341_large_left :=
  Budget.RowUpperBound.branch row9341_large_left_left_covers row9341_large_left_right_covers

def row9341_large_right_left : List Nat := row9341_block_009 ++ row9341_block_010
theorem row9341_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9341 row9341_large_right_left :=
  Budget.RowUpperBound.branch row9341_block_009_covers row9341_block_010_covers

def row9341_large_right_right_right : List Nat := row9341_block_012 ++ row9341_block_013
theorem row9341_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9341 row9341_large_right_right_right :=
  Budget.RowUpperBound.branch row9341_block_012_covers row9341_block_013_covers

def row9341_large_right_right : List Nat := row9341_block_011 ++ row9341_large_right_right_right
theorem row9341_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9341 row9341_large_right_right :=
  Budget.RowUpperBound.branch row9341_block_011_covers row9341_large_right_right_right_covers

def row9341_large_right : List Nat := row9341_large_right_left ++ row9341_large_right_right
theorem row9341_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9341 row9341_large_right :=
  Budget.RowUpperBound.branch row9341_large_right_left_covers row9341_large_right_right_covers

def row9341_large : List Nat := row9341_large_left ++ row9341_large_right
theorem row9341_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9341 row9341_large :=
  Budget.RowUpperBound.branch row9341_large_left_covers row9341_large_right_covers

def row9341 : List Nat := row9341_large
theorem row9341_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9341 row9341 := row9341_large_covers

def row9343_small : List Nat := []
theorem row9343_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9343 row9343_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9343_block_005 : List Nat := [9403, 9949]
def plan9343_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .possible, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9343_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9343 row9343_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9343_block_005
  · decide
  · rfl

def row9343_block_006 : List Nat := [11839]
def plan9343_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.secondFactor 29), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 17), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row9343_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9343 row9343_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9343_block_006
  · decide
  · rfl

def row9343_block_007 : List Nat := [13441, 14281]
def plan9343_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 19), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 19), .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .possible, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9343_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9343 row9343_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9343_block_007
  · decide
  · rfl

def row9343_block_008 : List Nat := [15013, 15073, 16057, 16417]
def plan9343_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9343_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9343 row9343_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9343_block_008
  · decide
  · rfl

def row9343_block_009 : List Nat := []
def plan9343_block_009 : List Budget.Choice := [(.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9343_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9343 row9343_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9343_block_009
  · decide
  · rfl

def row9343_block_010 : List Nat := []
theorem row9343_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9343 row9343_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9343_block_011 : List Nat := []
theorem row9343_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9343 row9343_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9343_block_012 : List Nat := []
theorem row9343_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9343 row9343_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9343_block_013 : List Nat := []
theorem row9343_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9343 row9343_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9343_large_left_left : List Nat := row9343_small ++ row9343_block_005
theorem row9343_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9343 row9343_large_left_left :=
  Budget.RowUpperBound.branch row9343_small_covers row9343_block_005_covers

def row9343_large_left_right_right : List Nat := row9343_block_007 ++ row9343_block_008
theorem row9343_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9343 row9343_large_left_right_right :=
  Budget.RowUpperBound.branch row9343_block_007_covers row9343_block_008_covers

def row9343_large_left_right : List Nat := row9343_block_006 ++ row9343_large_left_right_right
theorem row9343_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9343 row9343_large_left_right :=
  Budget.RowUpperBound.branch row9343_block_006_covers row9343_large_left_right_right_covers

def row9343_large_left : List Nat := row9343_large_left_left ++ row9343_large_left_right
theorem row9343_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9343 row9343_large_left :=
  Budget.RowUpperBound.branch row9343_large_left_left_covers row9343_large_left_right_covers

def row9343_large_right_left : List Nat := row9343_block_009 ++ row9343_block_010
theorem row9343_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9343 row9343_large_right_left :=
  Budget.RowUpperBound.branch row9343_block_009_covers row9343_block_010_covers

def row9343_large_right_right_right : List Nat := row9343_block_012 ++ row9343_block_013
theorem row9343_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9343 row9343_large_right_right_right :=
  Budget.RowUpperBound.branch row9343_block_012_covers row9343_block_013_covers

def row9343_large_right_right : List Nat := row9343_block_011 ++ row9343_large_right_right_right
theorem row9343_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9343 row9343_large_right_right :=
  Budget.RowUpperBound.branch row9343_block_011_covers row9343_large_right_right_right_covers

def row9343_large_right : List Nat := row9343_large_right_left ++ row9343_large_right_right
theorem row9343_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9343 row9343_large_right :=
  Budget.RowUpperBound.branch row9343_large_right_left_covers row9343_large_right_right_covers

def row9343_large : List Nat := row9343_large_left ++ row9343_large_right
theorem row9343_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9343 row9343_large :=
  Budget.RowUpperBound.branch row9343_large_left_covers row9343_large_right_covers

def row9343 : List Nat := row9343_large
theorem row9343_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9343 row9343 := row9343_large_covers

def row9349_small : List Nat := []
theorem row9349_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9349 row9349_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9349_block_005 : List Nat := [9619, 9661]
def plan9349_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .second, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row9349_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9349 row9349_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9349_block_005
  · decide
  · rfl

def row9349_block_006 : List Nat := [10399, 11239]
def plan9349_block_006 : List Budget.Choice := [.first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 29), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 19), .first, .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 13), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9349_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9349 row9349_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9349_block_006
  · decide
  · rfl

def row9349_block_007 : List Nat := [12703, 12799, 12829, 13219, 13381, 14221]
def plan9349_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .second, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 13), .possible, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 19), .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .possible, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9349_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9349 row9349_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9349_block_007
  · decide
  · rfl

def row9349_block_008 : List Nat := [14593, 14821, 15139, 15451, 15607, 15649, 15733, 15913]
def plan9349_block_008 : List Budget.Choice := [(.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .possible, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .possible, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 17), (.secondFactor 11), .possible, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9349_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9349 row9349_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9349_block_008
  · decide
  · rfl

def row9349_block_009 : List Nat := []
def plan9349_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9349_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9349 row9349_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9349_block_009
  · decide
  · rfl

def row9349_block_010 : List Nat := []
theorem row9349_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9349 row9349_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9349_block_011 : List Nat := []
theorem row9349_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9349 row9349_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9349_block_012 : List Nat := []
theorem row9349_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9349 row9349_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9349_block_013 : List Nat := []
theorem row9349_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9349 row9349_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9349_large_left_left : List Nat := row9349_small ++ row9349_block_005
theorem row9349_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9349 row9349_large_left_left :=
  Budget.RowUpperBound.branch row9349_small_covers row9349_block_005_covers

def row9349_large_left_right_right : List Nat := row9349_block_007 ++ row9349_block_008
theorem row9349_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9349 row9349_large_left_right_right :=
  Budget.RowUpperBound.branch row9349_block_007_covers row9349_block_008_covers

def row9349_large_left_right : List Nat := row9349_block_006 ++ row9349_large_left_right_right
theorem row9349_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9349 row9349_large_left_right :=
  Budget.RowUpperBound.branch row9349_block_006_covers row9349_large_left_right_right_covers

def row9349_large_left : List Nat := row9349_large_left_left ++ row9349_large_left_right
theorem row9349_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9349 row9349_large_left :=
  Budget.RowUpperBound.branch row9349_large_left_left_covers row9349_large_left_right_covers

def row9349_large_right_left : List Nat := row9349_block_009 ++ row9349_block_010
theorem row9349_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9349 row9349_large_right_left :=
  Budget.RowUpperBound.branch row9349_block_009_covers row9349_block_010_covers

def row9349_large_right_right_right : List Nat := row9349_block_012 ++ row9349_block_013
theorem row9349_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9349 row9349_large_right_right_right :=
  Budget.RowUpperBound.branch row9349_block_012_covers row9349_block_013_covers

def row9349_large_right_right : List Nat := row9349_block_011 ++ row9349_large_right_right_right
theorem row9349_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9349 row9349_large_right_right :=
  Budget.RowUpperBound.branch row9349_block_011_covers row9349_large_right_right_right_covers

def row9349_large_right : List Nat := row9349_large_right_left ++ row9349_large_right_right
theorem row9349_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9349 row9349_large_right :=
  Budget.RowUpperBound.branch row9349_large_right_left_covers row9349_large_right_right_covers

def row9349_large : List Nat := row9349_large_left ++ row9349_large_right
theorem row9349_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9349 row9349_large :=
  Budget.RowUpperBound.branch row9349_large_left_covers row9349_large_right_covers

def row9349 : List Nat := row9349_large
theorem row9349_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9349 row9349 := row9349_large_covers

end ProjectEulerSolutions.PrimePairLarge
