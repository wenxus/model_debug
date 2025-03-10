import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import _ from 'lodash';

const FeatureMatrixPlot = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [f0Selection, setF0Selection] = useState(null);
  const [questionSelection, setQuestionSelection] = useState('q1');
  const [f1Values, setF1Values] = useState([]);
  const [f2Values, setF2Values] = useState([]);
  const [matrices, setMatrices] = useState({});
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await window.fs.readFile('feature_steer_sweep_all_processed  feature_steer_sweep_all_processed.csv');
        const text = new TextDecoder().decode(response);
        
        Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const parsedData = results.data.filter(row => 
              row.f0 !== undefined && row.f1 !== undefined && row.f2 !== undefined
            );
            
            // Extract unique f0, f1, and f2 values
            const uniqueF0 = [...new Set(parsedData.map(row => row.f0))].sort((a, b) => a - b);
            const uniqueF1 = [...new Set(parsedData.map(row => row.f1))].sort((a, b) => a - b);
            const uniqueF2 = [...new Set(parsedData.map(row => row.f2))].sort((a, b) => a - b);
            
            setData(parsedData);
            setF1Values(uniqueF1);
            setF2Values(uniqueF2);
            setF0Selection(uniqueF0[0]); // Default to first f0 value
            
            // Create matrices for each f0 value
            const matrixResults = {};
            
            for (const f0 of uniqueF0) {
              matrixResults[f0] = {
                q1: createMatrix(parsedData, f0, 'answer_to_q1_is_correct', uniqueF1, uniqueF2),
                q2: createMatrix(parsedData, f0, 'answer_to_q2_is_correct', uniqueF1, uniqueF2)
              };
            }
            
            setMatrices(matrixResults);
            setLoading(false);
          },
          error: (error) => {
            console.error('Error parsing CSV:', error);
            setLoading(false);
          }
        });
      } catch (error) {
        console.error('Error reading file:', error);
        setLoading(false);
      }
    };
    
    // Helper function to create matrix for a specific f0 and question
    const createMatrix = (data, f0Value, questionKey, f1Values, f2Values) => {
      const matrix = [];
      for (let f1 of f1Values) {
        const row = [];
        for (let f2 of f2Values) {
          const point = data.find(d => 
            d.f0 === f0Value && 
            Math.abs(d.f1 - f1) < 0.0001 && 
            Math.abs(d.f2 - f2) < 0.0001
          );
          row.push(point ? point[questionKey] : null);
        }
        matrix.push(row);
      }
      return matrix;
    };
    
    fetchData();
  }, []);
  
  // Function to get cell color based on value (blue heatmap)
  const getCellColor = (value) => {
    if (value === null) return 'bg-blue-100'; // Missing values shown with same color as 0
    
    if (questionSelection === 'q1') {
      // Blue gradient for q1 values (0 to 1)
      if (value === 0) return 'bg-blue-100';
      if (value === 1) return 'bg-blue-700';
      // For any values between 0 and 1
      const intensity = Math.floor(value * 6);
      return `bg-blue-${100 + intensity * 100}`;
    } else {
      // Since all q2 values are 0, just use a single blue
      return 'bg-blue-100';
    }
  };
  
  // Function to get text color based on background color
  const getTextColor = (value) => {
    // All cells will have black text (since bg-blue-100 is light)
    return 'text-black';
  };
  
  if (loading) {
    return <div className="text-center p-8">Loading data...</div>;
  }
  
  // Get the currently selected matrix
  const currentMatrix = f0Selection !== null && matrices[f0Selection] 
    ? matrices[f0Selection][questionSelection] 
    : [];
  
  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">Feature Matrix Plot (Counting Letters vs Verify Letter Count)</h2>
      
      <div className="flex space-x-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">Need Clarification Feature:</label>
          <select 
            value={f0Selection || ''}
            onChange={(e) => setF0Selection(Number(e.target.value))}
            className="p-2 border rounded"
          >
            {Object.keys(matrices).map(f0 => {
              const numValue = Number(f0);
              return (
                <option key={f0} value={f0}>
                  {numValue === 2 ? "no steering" : "f0 = " + numValue}
                </option>
              );
            })}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Values to Display:</label>
          <select 
            value={questionSelection}
            onChange={(e) => setQuestionSelection(e.target.value)}
            className="p-2 border rounded"
          >
            <option value="q1">Answer to Q1 (has variation)</option>
            <option value="q2">Answer to Q2 (all zeros)</option>
          </select>
        </div>
      </div>
      
      <div className="overflow-auto">
        <div className="relative flex">
          {/* Top axis label */}
          <div className="w-32"></div> {/* Empty corner cell */}
          <div className="flex-1 text-center font-medium mb-2">Verify Letter Count Feature</div>
        </div>
        
        <div className="flex">
          {/* Left axis title */}
          <div className="w-16 flex items-center justify-center">
            <span className="font-medium transform -rotate-90 whitespace-nowrap text-center">Counting Letters Feature</span>
          </div>
          
          {/* Left axis labels + matrix */}
          <div className="flex">
            {/* Left axis labels (f1 values) */}
            <div className="w-16 flex flex-col">
              {f1Values.map((f1, index) => (
                <div key={index} className="h-12 flex items-center justify-end pr-2">
                  <span className="text-xs">{f1 === 2 ? "no steering" : f1.toFixed(2)}</span>
                </div>
              ))}
            </div>
            
            {/* Matrix cells with top axis labels */}
            <div>
              {/* Top axis values (f2 values) */}
              <div className="flex h-12">
                {f2Values.map((f2, index) => (
                  <div key={index} className="w-12 flex items-end justify-center pb-1">
                    <span className="text-xs transform -rotate-45 origin-bottom-left">{f2 === 2 ? "no s" : f2.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              
              {/* Matrix cells */}
              {currentMatrix.map((row, rowIndex) => (
                <div key={rowIndex} className="flex">
                  {row.map((cell, colIndex) => (
                    <div 
                      key={colIndex} 
                      className={`w-12 h-12 flex items-center justify-center border ${getCellColor(cell)} ${getTextColor(cell)}`}
                      title={`Counting Letters Feature=${f1Values[rowIndex] === 2 ? "no steering" : f1Values[rowIndex]}, Verify Letter Count Feature=${f2Values[colIndex] === 2 ? "no steering" : f2Values[colIndex]}, value=${cell}`}
                    >
                      {cell !== null ? cell : '0'}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">Legend</h3>
        <div className="flex gap-4">
          <div className="flex items-center">
            <div className="w-6 h-6 bg-blue-100 border mr-2"></div>
            <span>Value = 0</span>
          </div>
          <div className="flex items-center">
            <div className="w-6 h-6 bg-blue-700 border mr-2"></div>
            <span>Value = 1</span>
          </div>
          <div className="flex items-center">
            <div className="w-6 h-6 bg-gray-200 border mr-2"></div>
            <span>Missing data (displayed as 0)</span>
          </div>
        </div>
      </div>
      
      <div className="mt-6">
        <p><strong>Current Selection:</strong> Need Clarification Feature = {Number(f0Selection) === 2 ? "no steering" : f0Selection}, showing {questionSelection === 'q1' ? 'Answer to Q1' : 'Answer to Q2'}</p>
        <p className="text-sm text-gray-600 mt-2">
          The matrix shows the {questionSelection === 'q1' ? 'answer_to_q1_is_correct' : 'answer_to_q2_is_correct'} values 
          for each combination of Counting Letters Feature (vertical axis) and Verify Letter Count Feature (horizontal axis) at the selected Need Clarification Feature value.
        </p>
      </div>
    </div>
  );
};

export default FeatureMatrixPlot;
