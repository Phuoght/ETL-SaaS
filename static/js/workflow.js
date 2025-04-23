$(document).ready(function() {
    window.jsPlumbInitialized = false; // Biến toàn cục
    jsPlumb.ready(function() {
        if (typeof jsPlumb !== 'undefined') {
            jsPlumb.setContainer('canvas');
            window.jsPlumbInitialized = true;
            console.log('jsPlumb initialized successfully');
        } else {
            console.error('jsPlumb is not defined. Please check if jsPlumb library is loaded.');
        }
    });

    // Zoom functionality
    let zoomLevel = 1;
    const zoomStep = 0.1;
    const minZoom = 0.5;
    const maxZoom = 2;

    // Pan functionality
    let isPanning = false;
    let startX = 0;
    let startY = 0;
    let panX = 0;
    let panY = 0;

    // Variable to store the currently selected loop node
    let selectedLoopNodeId = null;

    // Object to store jsPlumb instances for each loop node
    const loopJsPlumbInstances = {};

    // Object to store files for each node
    const nodeFiles = {};

    function updateTransform() {
        const canvas = $('#canvas');
        const wrapper = $('#canvas-wrapper');
        
        let canvasWidth = canvas.width();
        let canvasHeight = canvas.height();
        let wrapperWidth = wrapper.width();
        let wrapperHeight = wrapper.height();

        canvasWidth = canvasWidth > 0 ? canvasWidth : 2000;
        canvasHeight = canvasHeight > 0 ? canvasHeight : 2000;
        wrapperWidth = wrapperWidth > 0 ? wrapperWidth : 800;
        wrapperHeight = wrapperHeight > 0 ? wrapperHeight : 600;

        const scaledCanvasWidth = canvasWidth * zoomLevel;
        const scaledCanvasHeight = canvasHeight * zoomLevel;

        let minPanX = -(scaledCanvasWidth - wrapperWidth);
        let maxPanX = 0;
        minPanX = isFinite(minPanX) ? minPanX : -1000;
        maxPanX = isFinite(maxPanX) ? maxPanX : 0;
        panX = Math.max(minPanX, Math.min(maxPanX, panX));

        let minPanY = -(scaledCanvasHeight - wrapperHeight);
        let maxPanY = 0;
        minPanY = isFinite(minPanY) ? minPanY : -1000;
        maxPanY = isFinite(maxPanY) ? maxPanY : 0;
        panY = Math.max(minPanY, Math.min(maxPanY, panY));

        panX = isFinite(panX) ? panX : 0;
        panY = isFinite(panY) ? panY : 0;

        canvas.css('transform', `translate(${panX}px, ${panY}px) scale(${zoomLevel})`);
        $('#zoom-level').text(`Zoom: ${(zoomLevel * 100).toFixed(0)}%`);
        if (jsPlumbInitialized) {
            jsPlumb.setZoom(zoomLevel);
            jsPlumb.repaintEverything();
        }
    }

    $('#zoom-in').click(function() {
        if (zoomLevel < maxZoom) {
            zoomLevel += zoomStep;
            updateTransform();
        }
    });

    $('#zoom-out').click(function() {
        if (zoomLevel > minZoom) {
            zoomLevel -= zoomStep;
            updateTransform();
        }
    });

    $('#canvas').on('wheel', function(event) {
        event.preventDefault();
        const delta = event.originalEvent.deltaY > 0 ? -zoomStep : zoomStep;
        const newZoomLevel = zoomLevel + delta;
        if (newZoomLevel >= minZoom && newZoomLevel <= maxZoom) {
            zoomLevel = newZoomLevel;
            updateTransform();
        }
    });

    $('#canvas').on('mousedown', function(event) {
        if (!$(event.target).closest('.node, .loop-child-node').length) {
            isPanning = true;
            startX = event.clientX - panX;
            startY = event.clientY - panY;
            $('#canvas').css('cursor', 'grabbing');
        }
    });

    $(document).on('mousemove', function(event) {
        if (isPanning) {
            panX = event.clientX - startX;
            panY = event.clientY - startY;
            updateTransform();
        }
    });

    $(document).on('mouseup', function() {
        if (isPanning) {
            isPanning = false;
            $('#canvas').css('cursor', 'grab');
        }
    });

    // Node configurations
    const nodeConfigs = {
        input: [
            { name: 'path', label: 'Tệp đầu vào', type: 'file', accept: '.csv,.xlsx,.xls,.json', required: true }
        ],
        output: [
            { name: 'path', label: 'Tệp đầu ra', type: 'text', placeholder: 'output.csv', required: true },
            { name: 'format', label: 'Định dạng đầu ra', type: 'select', options: ['csv', 'xlsx', 'json'], required: true }
        ],
        select_columns: [
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true }
        ],
        filter: [
            { name: 'condition', label: 'Điều kiện', type: 'text', placeholder: 'column1 > 100', required: true }
        ],
        python_script: [
            { name: 'code', label: 'Mã Python', type: 'textarea', placeholder: 'df["new_col"] = df["col1"] + df["col2"]', required: true }
        ],
        remove_nulls: [
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true }
        ],
        deduplicate: [
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true }
        ],
        replace_values: [
            { name: 'column', label: 'Cột', type: 'text', required: true },
            { name: 'old_value', label: 'Giá trị cũ', type: 'text', required: true },
            { name: 'new_value', label: 'Giá trị mới', type: 'text', required: true }
        ],
        join: [
            { name: 'path', label: 'Tệp Join', type: 'file', accept: '.csv,.xlsx,.xls,.json', required: true },
            { name: 'join_key', label: 'Khóa Join (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'key1,key2', required: true },
            { name: 'join_type', label: 'Kiểu Join', type: 'select', options: ['inner', 'left', 'right', 'outer'], required: true }
        ],
        aggregate: [
            { name: 'group_by', label: 'Group By (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true },
            { name: 'agg_column', label: 'Cột Aggregate', type: 'text', required: true },
            { name: 'agg_func', label: 'Hàm Aggregate', type: 'select', options: ['sum', 'mean', 'count', 'min', 'max'], required: true }
        ],
        pivot: [
            { name: 'pivot_index', label: 'Index (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true },
            { name: 'pivot_columns', label: 'Columns (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column3', required: true },
            { name: 'pivot_values', label: 'Values', type: 'text', required: true },
            { name: 'pivot_aggfunc', label: 'Hàm Aggregate', type: 'select', options: ['sum', 'mean', 'count'], required: true }
        ],
        drop_columns: [
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true }
        ],
        split_records: [
            { name: 'max_rows', label: 'Số dòng tối đa mỗi file', type: 'number', placeholder: '1000', required: true },
            { name: 'output_prefix', label: 'Tiền tố file đầu ra', type: 'text', placeholder: 'split_', required: true },
            { name: 'condition', label: 'Điều kiện (tùy chọn)', type: 'text', placeholder: 'column1 > 100' }
        ],
        merge_records: [
            { name: 'input_files', label: 'Tệp đầu vào', type: 'file', accept: '.csv,.xlsx,.xls,.json', multiple: true, required: true },
            { name: 'merge_type', label: 'Kiểu Merge', type: 'select', options: ['concat', 'merge'], required: true },
            { name: 'join_key', label: 'Khóa Join (nếu merge)', type: 'text', placeholder: 'key1,key2' }
        ],
        route_on_attribute: [
            { name: 'route_column', label: 'Cột Route', type: 'text', required: true },
            { name: 'routes', label: 'Routes (JSON)', type: 'textarea', placeholder: '{"value1": "file1.csv", "value2": "file2.csv"}', required: true }
        ],
        enrich_data: [
            { name: 'enrich_file', label: 'Tệp bổ sung', type: 'file', accept: '.csv,.xlsx,.xls,.json', required: true },
            { name: 'join_key', label: 'Khóa Join (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'key1,key2', required: true },
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true }
        ],
        convert_format: [
            { name: 'format', label: 'Định dạng', type: 'select', options: ['csv', 'xlsx', 'json'], required: true }
        ],
        replace_text: [
            { name: 'column', label: 'Cột', type: 'text', required: true },
            { name: 'pattern', label: 'Mẫu', type: 'text', required: true },
            { name: 'replacement', label: 'Thay thế', type: 'text', required: true }
        ],
        execute_sql: [
            { name: 'query', label: 'Truy vấn SQL', type: 'textarea', placeholder: 'SELECT * FROM df WHERE column1 > 100', required: true }
        ],
        branching: [
            { name: 'condition', label: 'Điều kiện', type: 'text', placeholder: 'column1 > 100', required: true }
        ],
        external_task_sensor: [
            { name: 'task_id', label: 'Task ID', type: 'text', required: true },
            { name: 'timeout', label: 'Timeout (giây)', type: 'number', placeholder: '300', required: true }
        ],
        email_notification: [
            { name: 'recipient', label: 'Người nhận', type: 'email', required: true },
            { name: 'subject', label: 'Chủ đề', type: 'text', required: true },
            { name: 'body', label: 'Nội dung', type: 'textarea', required: true }
        ],
        file_sensor: [
            { name: 'file_path', label: 'Đường dẫn tệp', type: 'text', required: true },
            { name: 'timeout', label: 'Timeout (giây)', type: 'number', placeholder: '300', required: true }
        ],
        data_quality_check: [
            { name: 'rules', label: 'Quy tắc (JSON)', type: 'textarea', placeholder: '{"no_nulls": ["column1"], "unique": ["column2"]}', required: true }
        ],
        dynamic_split: [
            { name: 'split_key', label: 'Cột Split', type: 'text', required: true },
            { name: 'output_prefix', label: 'Tiền tố file đầu ra', type: 'text', placeholder: 'split_', required: true }
        ],
        normalize_data: [
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true },
            { name: 'method', label: 'Phương pháp', type: 'select', options: ['min_max', 'z_score'], required: true }
        ],
        fuzzy_match: [
            { name: 'column', label: 'Cột nguồn', type: 'text', required: true },
            { name: 'reference_file', label: 'Tệp tham chiếu', type: 'file', accept: '.csv,.xlsx,.xls,.json', required: true },
            { name: 'reference_column', label: 'Cột tham chiếu', type: 'text', required: true },
            { name: 'output_column', label: 'Cột đầu ra', type: 'text', required: true },
            { name: 'threshold', label: 'Ngưỡng (0-100)', type: 'number', placeholder: '80', required: true }
        ],
        data_masking: [
            { name: 'columns', label: 'Cột (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true },
            { name: 'method', label: 'Phương pháp', type: 'select', options: ['hash', 'mask'], required: true }
        ],
        json_flatten: [
            { name: 'json_column', label: 'Cột JSON', type: 'text', required: true },
            { name: 'prefix', label: 'Tiền tố', type: 'text', placeholder: 'flattened_' }
        ],
        stream_filter: [
            { name: 'condition', label: 'Điều kiện', type: 'text', placeholder: 'column1 > 100', required: true }
        ],
        http_enrich: [
            { name: 'url', label: 'URL API', type: 'text', required: true },
            { name: 'key_column', label: 'Cột khóa', type: 'text', required: true },
            { name: 'output_columns', label: 'Cột đầu ra (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'col1,col2' }
        ],
        window_function: [
            { name: 'function', label: 'Hàm', type: 'select', options: ['rank', 'row_number', 'cumsum'], required: true },
            { name: 'partition_by', label: 'Partition By', type: 'text', required: true },
            { name: 'output_column', label: 'Cột đầu ra', type: 'text', required: true }
        ],
        cross_join: [
            { name: 'join_file', label: 'Tệp Join', type: 'file', accept: '.csv,.xlsx,.xls,.json', required: true }
        ],
        conditional_split: [
            { name: 'conditions', label: 'Điều kiện (JSON)', type: 'textarea', placeholder: '[{"condition": "column1 > 100", "output": "file1.csv"}, {"condition": "column1 <= 100", "output": "file2.csv"}]', required: true }
        ],
        aggregate_multiple: [
            { name: 'group_by', label: 'Group By (phân cách bởi dấu phẩy)', type: 'text', placeholder: 'column1,column2', required: true },
            { name: 'aggregations', label: 'Aggregations (JSON)', type: 'textarea', placeholder: '[{"column": "col1", "func": "sum", "output_column": "sum_col1"}, {"column": "col2", "func": "mean", "output_column": "mean_col2"}]', required: true }
        ],
        loop: [
            { name: 'condition', label: 'Điều kiện thoát (tùy chọn)', type: 'text', placeholder: 'len(df) > 0' },
            { name: 'max_iterations', label: 'Số lần lặp tối đa', type: 'number', placeholder: '10', required: true }
        ]
    };

    // Drag and Drop from Toolbox to Canvas
    $('.toolbox-item').draggable({
        helper: 'clone',
        revert: 'invalid',
        zIndex: 1000,
        start: function(event, ui) {
            ui.helper.css('transition', 'none');
        }
    });

    $('#canvas').droppable({
        accept: '.toolbox-item',
        drop: function(event, ui) {
            const type = ui.draggable.data('type');
            const id = 'node_' + new Date().getTime();

            const existingInputs = $('.node[data-type="input"]').length;
            const existingOutputs = $('.node[data-type="output"]').length;

            if (type === 'input' && existingInputs > 0) {
                alert('Chỉ được phép có 1 node Input trong workflow!');
                return;
            }
            if (type === 'output' && existingOutputs > 0) {
                alert('Chỉ được phép có 1 node Output trong workflow!');
                return;
            }

            const offsetX = event.offsetX;
            const offsetY = event.offsetY;

            let adjustedLeft = (offsetX / zoomLevel) - (panX / zoomLevel);
            let adjustedTop = (offsetY / zoomLevel) - (panY / zoomLevel);

            adjustedLeft = isFinite(adjustedLeft) ? adjustedLeft : 0;
            adjustedTop = isFinite(adjustedTop) ? adjustedTop : 0;
            adjustedLeft = Math.max(0, Math.min(adjustedLeft, 2000));
            adjustedTop = Math.max(0, Math.min(adjustedTop, 2000));

            const node = $('<div>')
                .addClass('node')
                .attr('id', id)
                .attr('data-type', type)
                .attr('data-config', '{}')
                .css({
                    left: adjustedLeft + 'px',
                    top: adjustedTop + 'px'
                });

            if (type === 'loop') {
                node.addClass('loop');
                const loopContent = $('<div>').addClass('loop-content').text('Thêm node vào đây');
                node.append(type.replace(/_/g, ' ').toUpperCase())
                    .append('<button class="delete-node" onclick="deleteNode(\'' + id + '\')">X</button>')
                    .append(loopContent);
                node.data('loop-nodes', []);
                const loopJsPlumb = jsPlumb.getInstance();
                loopJsPlumb.setContainer(loopContent);
                loopJsPlumbInstances[id] = loopJsPlumb;
                node.click(function() {
                    $('.node.loop').removeClass('selected-loop');
                    $(this).addClass('selected-loop');
                    selectedLoopNodeId = id;
                });
                loopJsPlumb.bind('connection', function(info) {
                    updateWorkflow();
                });
                loopJsPlumb.bind('connectionDetached', function(info) {
                    updateLoopConnections(id);
                    updateWorkflow();
                });
            } else {
                node.html(type.replace(/_/g, ' ').toUpperCase() +
                    '<button class="delete-node" onclick="deleteNode(\'' + id + '\')">X</button>' +
                    '<button class="add-to-loop" onclick="addToLoop(\'' + id + '\')">Add to Loop</button>');
            }

            $(this).append(node);

            if (jsPlumbInitialized) {
                jsPlumb.draggable(node, {
                    containment: '#canvas',
                    grid: [10, 10],
                    drag: function() {
                        jsPlumb.repaintEverything();
                    },
                    stop: function() {
                        updateWorkflow();
                    }
                });

                jsPlumb.addEndpoint(id, {
                    anchor: 'Right',
                    isSource: true,
                    isTarget: false,
                    connector: ['Bezier', { curviness: 50 }],
                    maxConnections: 1,
                    endpoint: 'Dot',
                    paintStyle: { fill: '#3498db', radius: 5 },
                    connectorStyle: { stroke: '#3498db', strokeWidth: 2 }
                });

                jsPlumb.addEndpoint(id, {
                    anchor: 'Left',
                    isSource: false,
                    isTarget: true,
                    maxConnections: 1,
                    endpoint: 'Dot',
                    paintStyle: { fill: '#3498db', radius: 5 }
                });
            }

            node.dblclick(function() {
                configureNode(id);
            });

            updateWorkflow();
        }
    });

    window.addToLoop = function(nodeId) {
        if (!selectedLoopNodeId) {
            alert('Vui lòng chọn một node Loop trước!');
            return;
        }

        const loopNode = $('#' + selectedLoopNodeId);
        const childNode = $('#' + nodeId);
        const loopContent = loopNode.find('.loop-content');
        const loopNodes = loopNode.data('loop-nodes') || [];
        const loopJsPlumb = loopJsPlumbInstances[selectedLoopNodeId];

        if (loopNode.attr('id') === nodeId) {
            alert('Không thể thêm node Loop vào chính nó!');
            return;
        }

        if (childNode.hasClass('loop')) {
            alert('Không thể thêm node Loop vào node Loop khác!');
            return;
        }

        if (!loopNodes.includes(nodeId)) {
            loopNodes.push(nodeId);
            loopNode.data('loop-nodes', loopNodes);
            if (loopNodes.length === 1) {
                loopContent.empty();
            }

            const childNodeDiv = $('<div>')
                .addClass('loop-child-node')
                .attr('id', 'loop-child-' + nodeId)
                .attr('data-type', childNode.attr('data-type'))
                .css({
                    top: (loopNodes.length - 1) * 60 + 'px',
                    left: '10px'
                })
                .html(childNode.attr('data-type').replace(/_/g, ' ').toUpperCase() +
                    '<button class="remove-from-loop" onclick="removeFromLoop(\'' + selectedLoopNodeId + '\', \'' + nodeId + '\')">X</button>');

            loopContent.append(childNodeDiv);

            loopJsPlumb.draggable(childNodeDiv, {
                containment: loopContent,
                grid: [5, 5],
                drag: function() {
                    loopJsPlumb.repaintEverything();
                },
                stop: function() {
                    updateLoopConnections(selectedLoopNodeId);
                    updateWorkflow();
                }
            });

            loopJsPlumb.addEndpoint('loop-child-' + nodeId, {
                anchor: 'Right',
                isSource: true,
                isTarget: false,
                connector: ['Bezier', { curviness: 20 }],
                maxConnections: 1,
                endpoint: 'Dot',
                paintStyle: { fill: '#e74c3c', radius: 3 },
                connectorStyle: { stroke: '#e74c3c', strokeWidth: 1 }
            });

            loopJsPlumb.addEndpoint('loop-child-' + nodeId, {
                anchor: 'Left',
                isSource: false,
                isTarget: true,
                maxConnections: 1,
                endpoint: 'Dot',
                paintStyle: { fill: '#e74c3c', radius: 3 }
            });

            childNode.hide();
            if (jsPlumbInitialized) {
                jsPlumb.removeAllEndpoints(nodeId);
            }
            updateWorkflow();
        }
    };

    window.removeFromLoop = function(loopId, nodeId) {
        const loopNode = $('#' + loopId);
        const loopContent = loopNode.find('.loop-content');
        const loopNodes = loopNode.data('loop-nodes') || [];
        const loopJsPlumb = loopJsPlumbInstances[loopId];

        const index = loopNodes.indexOf(nodeId);
        if (index !== -1) {
            loopNodes.splice(index, 1);
            loopNode.data('loop-nodes', loopNodes);
            $('#loop-child-' + nodeId).remove();
            loopJsPlumb.remove('loop-child-' + nodeId);

            const childNode = $('#' + nodeId);
            childNode.show();
            if (jsPlumbInitialized) {
                jsPlumb.addEndpoint(nodeId, {
                    anchor: 'Right',
                    isSource: true,
                    isTarget: false,
                    connector: ['Bezier', { curviness: 50 }],
                    maxConnections: 1,
                    endpoint: 'Dot',
                    paintStyle: { fill: '#3498db', radius: 5 },
                    connectorStyle: { stroke: '#3498db', strokeWidth: 2 }
                });
                jsPlumb.addEndpoint(nodeId, {
                    anchor: 'Left',
                    isSource: false,
                    isTarget: true,
                    maxConnections: 1,
                    endpoint: 'Dot',
                    paintStyle: { fill: '#3498db', radius: 5 }
                });
            }

            if (loopNodes.length === 0) {
                loopContent.text('Thêm node vào đây');
            }

            updateLoopConnections(loopId);
            updateWorkflow();
        }
    };

    function updateLoopConnections(loopId) {
        const loopJsPlumb = loopJsPlumbInstances[loopId];
        const loopConnections = loopJsPlumb.getConnections().map(function(conn) {
            return {
                source: conn.sourceId.replace('loop-child-', ''),
                target: conn.targetId.replace('loop-child-', '')
            };
        });
        $('#' + loopId).data('loop-connections', loopConnections);
    }

    if (jsPlumbInitialized) {
        jsPlumb.bind('connection', function(info) {
            const sourceId = info.sourceId;
            const targetId = info.targetId;
            const sourceType = $('#' + sourceId).data('type');
            const targetType = $('#' + targetId).data('type');

            if (sourceType === 'input' && jsPlumb.getConnections({ target: sourceId }).length > 0) {
                jsPlumb.deleteConnection(info.connection);
                alert('Node Input không thể có kết nối đi vào!');
                return;
            }
            if (targetType === 'output' && jsPlumb.getConnections({ source: targetId }).length > 0) {
                jsPlumb.deleteConnection(info.connection);
                alert('Node Output không thể có kết nối đi ra!');
                return;
            }

            updateWorkflow();
        });

        jsPlumb.bind('connectionDetached', function(info) {
            updateWorkflow();
        });
    }

    window.deleteNode = function(nodeId) {
        if (confirm('Bạn có chắc chắn muốn xóa node này?')) {
            const node = $('#' + nodeId);
            const type = node.data('type');
            if (type === 'loop') {
                const loopNodes = node.data('loop-nodes') || [];
                loopNodes.forEach(childId => {
                    const childNode = $('#' + childId);
                    childNode.show();
                    if (jsPlumbInitialized) {
                        jsPlumb.addEndpoint(childId, {
                            anchor: 'Right',
                            isSource: true,
                            isTarget: false,
                            connector: ['Bezier', { curviness: 50 }],
                            maxConnections: 1,
                            endpoint: 'Dot',
                            paintStyle: { fill: '#3498db', radius: 5 },
                            connectorStyle: { stroke: '#3498db', strokeWidth: 2 }
                        });
                        jsPlumb.addEndpoint(childId, {
                            anchor: 'Left',
                            isSource: false,
                            isTarget: true,
                            maxConnections: 1,
                            endpoint: 'Dot',
                            paintStyle: { fill: '#3498db', radius: 5 }
                        });
                    }
                });
                if (selectedLoopNodeId === nodeId) {
                    selectedLoopNodeId = null;
                    $('.node.loop').removeClass('selected-loop');
                }
                delete loopJsPlumbInstances[nodeId];
            }
            delete nodeFiles[nodeId];
            if (jsPlumbInitialized) {
                jsPlumb.remove(nodeId);
            }
            updateWorkflow();
        }
    };

    function validateConfig(type, formData, nodeId) {
        const errors = [];
        const config = nodeConfigs[type] || [];

        config.forEach(field => {
            let value = formData[field.name];

            if (field.type === 'file') {
                const fileInput = $(`#config-${field.name}`);
                const files = fileInput[0]?.files;

                if (files && files.length > 0) {
                    if (field.multiple) {
                        formData[field.name] = Array.from(files).map(f => f.name).join(',');
                    } else {
                        formData[field.name] = files[0].name;
                    }
                    value = formData[field.name];
                } else if (formData[field.name]) {
                    value = formData[field.name];
                }

                if (field.required && !value) {
                    errors.push(`Vui lòng chọn ${field.label}.`);
                }
            } else if (field.required && (!value || value.trim() === '')) {
                errors.push(`Vui lòng nhập ${field.label}.`);
            } else if (field.type === 'number' && value && (isNaN(value) || parseFloat(value) <= 0)) {
                errors.push(`${field.label} phải là số dương hợp lệ.`);
            } else if (field.name === 'routes' || field.name === 'rules' || field.name === 'conditions' || field.name === 'aggregations') {
                if (value) {
                    try {
                        const parsed = JSON.parse(value);
                        if (field.name === 'routes' && (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))) {
                            errors.push(`${field.label} phải là một object JSON hợp lệ (e.g., {"value1": "file1.csv"}).`);
                        }
                        if (field.name === 'rules' && (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))) {
                            errors.push(`${field.label} phải là một object JSON hợp lệ (e.g., {"no_nulls": ["col1"]}).`);
                        }
                        if (field.name === 'conditions' && (!Array.isArray(parsed) || !parsed.every(item => item.condition && item.output))) {
                            errors.push(`${field.label} phải là một mảng JSON hợp lệ (e.g., [{"condition": "col1 > 100", "output": "file.csv"}]).`);
                        }
                        if (field.name === 'aggregations' && (!Array.isArray(parsed) || !parsed.every(item => item.column && item.func && item.output_column))) {
                            errors.push(`${field.label} phải là một mảng JSON hợp lệ (e.g., [{"column": "col1", "func": "sum", "output_column": "sum_col1"}]).`);
                        }
                    } catch (e) {
                        errors.push(`${field.label} phải là JSON hợp lệ.`);
                    }
                }
            } else if (field.name === 'threshold' && value && (isNaN(value) || parseFloat(value) < 0 || parseFloat(value) > 100)) {
                errors.push(`${field.label} phải từ 0 đến 100.`);
            } else if (field.type === 'email' && value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
                errors.push(`${field.label} phải là email hợp lệ.`);
            } else if (field.name === 'columns' || field.name === 'group_by' || field.name === 'join_key' || field.name === 'pivot_index' || field.name === 'pivot_columns' || field.name === 'output_columns') {
                if (value) {
                    const cols = value.split(',').map(c => c.trim());
                    if (!cols.every(c => c.length > 0)) {
                        errors.push(`${field.label} không được chứa phần tử trống.`);
                    }
                }
            } else if (field.name === 'format' && value && !['csv', 'xlsx', 'json'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: csv, xlsx, json.`);
            } else if (field.name === 'join_type' && value && !['inner', 'left', 'right', 'outer'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: inner, left, right, outer.`);
            } else if (field.name === 'agg_func' && value && !['sum', 'mean', 'count', 'min', 'max'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: sum, mean, count, min, max.`);
            } else if (field.name === 'pivot_aggfunc' && value && !['sum', 'mean', 'count'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: sum, mean, count.`);
            } else if (field.name === 'merge_type' && value && !['concat', 'merge'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: concat, merge.`);
            } else if (field.name === 'method' && value && !['min_max', 'z_score', 'hash', 'mask'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: ${field.options.join(', ')}.`);
            } else if (field.name === 'function' && value && !['rank', 'row_number', 'cumsum'].includes(value)) {
                errors.push(`${field.label} phải là một trong các giá trị: rank, row_number, cumsum.`);
            }
        });

        if (type === 'merge_records' && formData['merge_type'] === 'merge' && (!formData['join_key'] || formData['join_key'].trim() === '')) {
            errors.push('Vui lòng nhập Khóa Join khi chọn kiểu Merge.');
        }

        if (type === 'output') {
            const filePath = formData['path'];
            const format = formData['format'];

            const extension = filePath.split('.').pop().toLowerCase();
            const hasExtension = filePath.includes('.') && extension.length > 0;

            if (!hasExtension) {
                errors.push('Tệp đầu ra phải có đuôi file (ví dụ: .csv, .xlsx, .json).');
            } else {
                const expectedExtension = format.toLowerCase();
                if (extension !== expectedExtension) {
                    errors.push(`Đuôi file (${extension}) không khớp với định dạng đầu ra (${format}).`);
                }
            }
        }

        return errors;
    }

    function configureNode(nodeId) {
        const node = $('#' + nodeId);
        const type = node.data('type');
        const config = JSON.parse(node.attr('data-config') || '{}');
        $('#modal-title').text('Cấu hình ' + type.replace(/_/g, ' ').toUpperCase());
        $('#config-fields').empty();
        $('#config-fields').append('<div id="config-errors" class="notification error" style="display: none;"></div>');

        const fields = nodeConfigs[type] || [];
        fields.forEach(field => {
            const fieldId = `config-${field.name}`;
            let inputHtml = '';
            let currentValueDisplay = '';

            if (field.type === 'file') {
                const currentFile = config[field.name] ? `Đã chọn: ${config[field.name]}` : '';
                inputHtml = `
                    <input type="file" id="${fieldId}" name="node_${nodeId}_file" ${field.accept ? `accept="${field.accept}"` : ''} ${field.multiple ? 'multiple' : ''} ${field.required ? 'required' : ''}>
                    <p class="current-file">${currentFile}</p>
                `;
            } else if (field.type === 'select') {
                inputHtml = `<select id="${fieldId}" name="${field.name}" ${field.required ? 'required' : ''}>`;
                field.options.forEach(opt => {
                    inputHtml += `<option value="${opt}" ${config[field.name] === opt ? 'selected' : ''}>${opt}</option>`;
                });
                inputHtml += '</select>';
            } else if (field.type === 'textarea') {
                const currentValue = config[field.name] ? `Hiện tại:\n${config[field.name]}` : '';
                inputHtml = `<textarea id="${fieldId}" name="${field.name}" placeholder="${field.placeholder || ''}" ${field.required ? 'required' : ''}>${config[field.name] || ''}</textarea>`;
                currentValueDisplay = `<p class="current-value">${currentValue}</p>`;
            } else {
                inputHtml = `<input type="${field.type}" id="${fieldId}" name="${field.name}" placeholder="${field.placeholder || ''}" value="${config[field.name] || ''}" ${field.required ? 'required' : ''}>`;
            }

            $('#config-fields').append(`
                <div class="form-group">
                    <label for="${fieldId}">${field.label}</label>
                    ${inputHtml}
                    ${currentValueDisplay}
                </div>
            `);
        });

        $('#config-modal').addClass('show animate__fadeInDown');

        $('#save-config').off('click').on('click', function() {
            const formData = {};
            $('#config-fields').find('input, select, textarea').each(function() {
                const name = $(this).attr('name');
                if (name && !name.includes('file')) {
                    formData[name] = $(this).val();
                } else if (name && name.includes('file') && $(this)[0].files.length) {
                    if ($(this)[0].hasAttribute('multiple')) {
                        formData[name] = Array.from($(this)[0].files).map(f => f.name).join(',');
                        nodeFiles[nodeId] = Array.from($(this)[0].files);
                    } else {
                        formData[name] = $(this)[0].files[0].name;
                        nodeFiles[nodeId] = $(this)[0].files[0];
                    }
                }
            });

            fields.forEach(field => {
                if (field.type === 'file' && !formData[field.name] && config[field.name]) {
                    formData[field.name] = config[field.name];
                }
            });

            const errors = validateConfig(type, formData, nodeId);
            if (errors.length > 0) {
                $('#config-errors').text(errors.join(' ')).show();
                return;
            }

            if (type === 'loop') {
                formData.loop_nodes = node.data('loop-nodes') || [];
                formData.loop_connections = node.data('loop-connections') || [];
            }

            node.attr('data-config', JSON.stringify(formData));
            $('#config-modal').removeClass('show animate__fadeInDown').addClass('animate__fadeOutUp');
            setTimeout(() => $('#config-modal').removeClass('animate__fadeOutUp'), 300);
            updateWorkflow();
        });

        $('#cancel-config').off('click').on('click', function() {
            $('#config-modal').removeClass('show animate__fadeInDown').addClass('animate__fadeOutUp');
            setTimeout(() => $('#config-modal').removeClass('animate__fadeOutUp'), 300);
        });
    }

    function validateFlowStructure(nodes, connections) {
        const inputNodes = nodes.filter(n => n.type === 'input');
        const outputNodes = nodes.filter(n => n.type === 'output');

        if (inputNodes.length !== 1) {
            return `Flow phải có đúng 1 node Input! Hiện tại có ${inputNodes.length} node Input: ${inputNodes.map(n => n.id).join(', ')}.`;
        }
        if (outputNodes.length !== 1) {
            return `Flow phải có đúng 1 node Output! Hiện tại có ${outputNodes.length} node Output: ${outputNodes.map(n => n.id).join(', ')}.`;
        }

        const inputNodeId = inputNodes[0].id;
        const outputNodeId = outputNodes[0].id;

        const inputIncoming = connections.filter(conn => conn.target === inputNodeId);
        if (inputIncoming.length > 0) {
            return `Node Input (${inputNodeId}) không được có kết nối đi vào!`;
        }

        const outputOutgoing = connections.filter(conn => conn.source === outputNodeId);
        if (outputOutgoing.length > 0) {
            return `Node Output (${outputNodeId}) không được có kết nối đi ra!`;
        }

        const intermediateNodes = nodes.filter(n => n.type !== 'input' && n.type !== 'output');
        for (const node of intermediateNodes) {
            const incoming = connections.filter(conn => conn.target === node.id);
            const outgoing = connections.filter(conn => conn.source === node.id);
            if (node.type !== 'loop') {
                if (incoming.length !== 1) {
                    return `Node ${node.id} (${node.type}) phải có đúng 1 kết nối đi vào! Hiện tại có ${incoming.length} kết nối đi vào.`;
                }
                if (outgoing.length !== 1) {
                    return `Node ${node.id} (${node.type}) phải có đúng 1 kết nối đi ra! Hiện tại có ${outgoing.length} kết nối đi ra.`;
                }
            } else {
                const loopNodes = node.config.loop_nodes || [];
                const loopConnections = node.config.loop_connections || [];
                if (loopNodes.length > 0) {
                    const loopIntermediateNodes = loopNodes.map(id => nodes.find(n => n.id === id));
                    for (const ln of loopIntermediateNodes) {
                        const lnIncoming = loopConnections.filter(conn => conn.target === ln.id);
                        const lnOutgoing = loopConnections.filter(conn => conn.source === ln.id);
                        if (lnIncoming.length > 1) {
                            return `Node ${ln.id} (${ln.type}) trong loop ${node.id} có nhiều hơn 1 kết nối đi vào!`;
                        }
                        if (lnOutgoing.length > 1) {
                            return `Node ${ln.id} (${ln.type}) trong loop ${node.id} có nhiều hơn 1 kết nối đi ra!`;
                        }
                    }
                }
            }
        }

        console.log('Connections:', connections);

        const visited = new Set();
        function dfs(currentId, targetId) {
            if (currentId === targetId) {
                visited.add(currentId);
                return true;
            }
            if (visited.has(currentId)) return false;
            visited.add(currentId);

            const node = nodes.find(n => n.id === currentId);
            if (node.type === 'loop') {
                const loopNodes = node.config.loop_nodes || [];
                if (!loopNodes.length) {
                    return false;
                }
                for (const loopNodeId of loopNodes) {
                    if (!nodes.find(n => n.id === loopNodeId)) {
                        return false;
                    }
                }
                const loopConnections = node.config.loop_connections || [];
                const loopVisited = new Set();
                function loopDfs(loopCurrentId, loopTargetId) {
                    if (loopCurrentId === loopTargetId) {
                        loopVisited.add(loopCurrentId);
                        return true;
                    }
                    if (loopVisited.has(loopCurrentId)) return false;
                    loopVisited.add(loopCurrentId);

                    const loopOutgoing = loopConnections.filter(conn => conn.source === loopCurrentId);
                    for (const conn of loopOutgoing) {
                        if (loopDfs(conn.target, loopTargetId)) {
                            loopVisited.add(conn.target);
                            return true;
                        }
                    }
                    return false;
                }

                const loopStartNode = loopNodes[0];
                for (const loopNodeId of loopNodes) {
                    loopVisited.clear();
                    if (loopDfs(loopStartNode, loopNodeId)) {
                        visited.add(loopNodeId);
                    }
                }
            }

            const outgoing = connections.filter(conn => conn.source === currentId);
            for (const conn of outgoing) {
                if (dfs(conn.target, targetId)) {
                    visited.add(conn.target);
                    return true;
                }
            }
            return false;
        }

        const pathExists = dfs(inputNodeId, outputNodeId);
        if (!pathExists) {
            return `Không có đường dẫn từ Input (${inputNodeId}) đến Output (${outputNodeId})! Vui lòng kiểm tra các kết nối.`;
        }

        console.log('Visited nodes:', Array.from(visited));

        const allNodeIds = nodes.map(n => n.id);
        const unusedNodes = allNodeIds.filter(id => !visited.has(id));
        for (const nodeId of unusedNodes) {
            const node = nodes.find(n => n.id === nodeId);
            if (nodeId === inputNodeId || nodeId === outputNodeId) {
                continue;
            }
            if (node.type !== 'loop') {
                return `Node ${nodeId} (${node.type}) không nằm trong đường dẫn từ Input (${inputNodeId}) đến Output (${outputNodeId})!`;
            }
            const loopNodes = node.config.loop_nodes || [];
            for (const loopNodeId of loopNodes) {
                if (!allNodeIds.includes(loopNodeId)) {
                    return `Node ${loopNodeId} trong loop_nodes của node Loop ${nodeId} không tồn tại!`;
                }
            }
        }

        return null;
    }

    function validateAllNodes(nodes) {
        const errors = [];
        nodes.forEach(node => {
            const type = node.type;
            const config = node.config || {};
            const nodeId = node.id;

            const fields = nodeConfigs[type] || [];
            fields.forEach(field => {
                const value = config[field.name];
                if (field.required && (!value || value.trim() === '')) {
                    if (field.type === 'file') {
                        errors.push(`Vui lòng tải lên tệp cho node ${type} ${nodeId}!`);
                    } else {
                        errors.push(`Vui lòng nhập ${field.label} cho node ${type} ${nodeId}!`);
                    }
                }
            });

            if (type === 'input' || fields.some(field => field.type === 'file')) {
                if (!nodeFiles[nodeId] && config.path) {
                    errors.push(`Tệp cho node ${type} ${nodeId} không được tải lên! Vui lòng chọn lại tệp.`);
                }
            }
        });
        return errors.length > 0 ? errors.join(' ') : null;
    }

    function updateWorkflow() {
        const nodes = [];
        $('.node').each(function() {
            const node = $(this);
            const configStr = node.attr('data-config') || '{}';
            const config = JSON.parse(configStr);
            if (node.data('type') === 'loop') {
                config.loop_nodes = node.data('loop-nodes') || [];
                config.loop_connections = node.data('loop-connections') || [];
            }
            nodes.push({
                id: node.attr('id'),
                type: node.data('type'),
                config: config,
                position: {
                    x: (parseInt(node.css('left')) + panX / zoomLevel) * zoomLevel,
                    y: (parseInt(node.css('top')) + panY / zoomLevel) * zoomLevel
                }
            });
        });

        let connections = [];
        if (jsPlumbInitialized) {
            connections = jsPlumb.getConnections().map(function(conn) {
                return {
                    source: conn.sourceId,
                    target: conn.targetId
                };
            });
        }

        const workflow = {
            nodes: nodes,
            connections: connections
        };

        $('#flow').val(JSON.stringify(workflow));
    }

    window.loadWorkflow = function(flow, viewOnly) {
        if (!window.jsPlumbInitialized) {
            console.error('jsPlumb is not initialized. Cannot load workflow.');
            alert('Không thể tải workflow vì jsPlumb chưa được khởi tạo. Vui lòng kiểm tra lại!');
            return;
        }
    
        if ($('#canvas').length === 0) {
            console.error('Canvas element not found in DOM.');
            alert('Không tìm thấy phần tử canvas trong DOM. Vui lòng kiểm tra lại HTML!');
            return;
        }
    
        if (!flow || !flow.nodes || !Array.isArray(flow.nodes)) {
            console.error('Flow không chứa nodes hoặc nodes không phải là mảng:', flow);
            alert('Workflow không chứa node nào hoặc dữ liệu không hợp lệ!');
            return;
        }
    
        if (!flow.connections || !Array.isArray(flow.connections)) {
            console.warn('Flow không chứa connections hoặc connections không phải là mảng:', flow.connections);
            flow.connections = [];
        }
    
        // Reset zoom và vị trí
        zoomLevel = 1;
        panX = 0;
        panY = 0;
    
        // Tối ưu: Chỉ xóa các node không còn trong flow mới
        const currentNodeIds = new Set($('.node').map(function() { return $(this).attr('id'); }).get());
        const newNodeIds = new Set(flow.nodes.map(node => node.id));
        currentNodeIds.forEach(nodeId => {
            if (!newNodeIds.has(nodeId)) {
                const node = $('#' + nodeId);
                const type = node.data('type');
                if (type === 'loop') {
                    const loopNodes = node.data('loop-nodes') || [];
                    loopNodes.forEach(childId => {
                        $('#' + childId).show();
                    });
                    delete loopJsPlumbInstances[nodeId];
                }
                jsPlumb.remove(nodeId);
            }
        });
    
        // Xóa tất cả kết nối hiện tại
        try {
            if (jsPlumb.deleteEveryConnection && jsPlumb.deleteEveryEndpoint) {
                jsPlumb.deleteEveryConnection();
                jsPlumb.deleteEveryEndpoint();
            } else {
                console.warn('jsPlumb methods for clearing connections/endpoints are not available.');
            }
        } catch (e) {
            console.error('Lỗi khi xóa kết nối và endpoint:', e);
        }
    
        jsPlumb.setContainer('canvas');
    
        const workflow = typeof flow === 'string' ? JSON.parse(flow) : flow;
        console.log('Loading workflow:', workflow);
    
        // Tạo hoặc cập nhật các node
        workflow.nodes.forEach(function(node) {
            if (!node.id || typeof node.id !== 'string') {
                console.error('Node không có id hợp lệ:', node);
                return;
            }
    
            if (!node.position || typeof node.position.x !== 'number' || typeof node.position.y !== 'number') {
                console.warn('Node position không hợp lệ, đặt mặc định:', node);
                node.position = { x: 0, y: 0 };
            }
    
            let adjustedLeft = node.position.x;
            let adjustedTop = node.position.y;
    
            adjustedLeft = isFinite(adjustedLeft) ? adjustedLeft : 0;
            adjustedTop = isFinite(adjustedTop) ? adjustedTop : 0;
            adjustedLeft = Math.max(0, Math.min(adjustedLeft, 2000));
            adjustedTop = Math.max(0, Math.min(adjustedTop, 2000));
    
            let div = $('#' + node.id);
            if (div.length === 0) {
                // Tạo mới node nếu chưa tồn tại
                div = $('<div>')
                    .addClass('node')
                    .attr('id', node.id)
                    .attr('data-type', node.type)
                    .attr('data-config', JSON.stringify(node.config || {}))
                    .css({
                        left: adjustedLeft + 'px',
                        top: adjustedTop + 'px'
                    });
                $('#canvas').append(div);
                console.log('Node appended to DOM:', node.id);
            } else {
                // Cập nhật vị trí và config cho node đã tồn tại
                div.css({
                    left: adjustedLeft + 'px',
                    top: adjustedTop + 'px'
                }).attr('data-type', node.type)
                  .attr('data-config', JSON.stringify(node.config || {}));
                console.log('Node updated in DOM:', node.id);
            }
    
            if (node.type === 'loop') {
                div.addClass('loop');
                let loopContent = div.find('.loop-content');
                if (loopContent.length === 0) {
                    loopContent = $('<div>').addClass('loop-content').text('Thêm node vào đây');
                    div.empty().append(node.type.replace(/_/g, ' ').toUpperCase());
                    if (!viewOnly) {
                        div.append('<button class="delete-node" onclick="deleteNode(\'' + node.id + '\')">X</button>');
                        div.off('click').on('click', function() {
                            $('.node.loop').removeClass('selected-loop');
                            $(this).addClass('selected-loop');
                            selectedLoopNodeId = node.id;
                        });
                    }
                    div.append(loopContent);
                }
                div.data('loop-nodes', node.config.loop_nodes || []);
                div.data('loop-connections', node.config.loop_connections || []);
    
                let loopJsPlumb = loopJsPlumbInstances[node.id];
                if (!loopJsPlumb) {
                    loopJsPlumb = jsPlumb.getInstance();
                    loopJsPlumb.setContainer(loopContent);
                    loopJsPlumbInstances[node.id] = loopJsPlumb;
    
                    if (!viewOnly) {
                        loopJsPlumb.bind('connection', function(info) {
                            updateLoopConnections(node.id);
                            updateWorkflow();
                        });
                        loopJsPlumb.bind('connectionDetached', function(info) {
                            updateLoopConnections(node.id);
                            updateWorkflow();
                        });
                    }
                }
    
                if (node.config.loop_nodes && node.config.loop_nodes.length > 0) {
                    loopContent.empty();
                    node.config.loop_nodes.forEach((childId, index) => {
                        const childNode = workflow.nodes.find(n => n.id === childId);
                        if (childNode) {
                            let childNodeDiv = $('#loop-child-' + childId);
                            if (childNodeDiv.length === 0) {
                                childNodeDiv = $('<div>')
                                    .addClass('loop-child-node')
                                    .attr('id', 'loop-child-' + childId)
                                    .attr('data-type', childNode.type)
                                    .css({
                                        top: index * 60 + 'px',
                                        left: '10px'
                                    })
                                    .html(childNode.type.replace(/_/g, ' ').toUpperCase() +
                                        (viewOnly ? '' : '<button class="remove-from-loop" onclick="removeFromLoop(\'' + node.id + '\', \'' + childId + '\')">X</button>'));
                                loopContent.append(childNodeDiv);
                            } else {
                                childNodeDiv.css({
                                    top: index * 60 + 'px',
                                    left: '10px'
                                }).attr('data-type', childNode.type);
                            }
    
                            if (!viewOnly) {
                                loopJsPlumb.draggable(childNodeDiv, {
                                    containment: loopContent,
                                    grid: [5, 5],
                                    drag: function() {
                                        loopJsPlumb.repaintEverything();
                                    },
                                    stop: function() {
                                        updateLoopConnections(node.id);
                                        updateWorkflow();
                                    }
                                });
                            }
    
                            if (childNodeDiv.find('.endpoint').length === 0) {
                                loopJsPlumb.addEndpoint('loop-child-' + childId, {
                                    anchor: 'Right',
                                    isSource: !viewOnly,
                                    isTarget: false,
                                    connector: ['Flowchart', { cornerRadius: 10, stub: [10, 15] }], // Sử dụng Flowchart thay vì Bezier
                                    maxConnections: 1,
                                    endpoint: 'Rectangle', // Đổi sang hình chữ nhật nhỏ
                                    paintStyle: { fill: '#1abc9c', width: 6, height: 6 },
                                    connectorStyle: { 
                                        stroke: '#1abc9c', // Màu xanh ngọc
                                        strokeWidth: 1.5, // Tăng độ dày
                                        dropShadow: { dx: 1, dy: 1, blur: 2, color: 'rgba(0,0,0,0.3)' } // Thêm bóng đổ
                                    },
                                    connectorOverlays: [
                                        ['Label', { 
                                            label: 'Loop Flow', 
                                            location: 0.5, 
                                            cssClass: 'jtk-overlay' 
                                        }]
                                    ]
                                });
    
                                loopJsPlumb.addEndpoint('loop-child-' + childId, {
                                    anchor: 'Left',
                                    isSource: false,
                                    isTarget: !viewOnly,
                                    maxConnections: 1,
                                    endpoint: 'Rectangle',
                                    paintStyle: { fill: '#1abc9c', width: 6, height: 6 }
                                });
                            }
    
                            const childElement = $('#' + childId);
                            if (childElement.length > 0) {
                                childElement.hide();
                            } else {
                                console.warn(`Child node ${childId} not found in DOM, cannot hide.`);
                            }
                        } else {
                            console.warn(`Child node ${childId} not found in workflow nodes`);
                        }
                    });
    
                    (node.config.loop_connections || []).forEach(conn => {
                        console.log('Rendering loop connection:', conn);
                        const existingConnections = loopJsPlumb.getConnections({
                            source: 'loop-child-' + conn.source,
                            target: 'loop-child-' + conn.target
                        });
                        if (existingConnections.length === 0) {
                            loopJsPlumb.connect({
                                source: 'loop-child-' + conn.source,
                                target: 'loop-child-' + conn.target,
                                connector: ['Flowchart', { cornerRadius: 10, stub: [10, 15] }],
                                endpoint: 'Rectangle',
                                paintStyle: { fill: '#1abc9c', width: 6, height: 6 },
                                connectorStyle: { 
                                    stroke: '#1abc9c',
                                    strokeWidth: 1.5,
                                    dropShadow: { dx: 1, dy: 1, blur: 2, color: 'rgba(0,0,0,0.3)' }
                                },
                                connectorOverlays: [
                                    ['Label', { 
                                        label: 'Loop Flow', 
                                        location: 0.5, 
                                        cssClass: 'jtk-overlay' 
                                    }]
                                ]
                            });
                        }
                    });
                }
            } else {
                if (div.find('.delete-node').length === 0) {
                    div.html(node.type.replace(/_/g, ' ').toUpperCase() +
                        (viewOnly ? '' : '<button class="delete-node" onclick="deleteNode(\'' + node.id + '\')">X</button>' +
                        '<button class="add-to-loop" onclick="addToLoop(\'' + node.id + '\')">Add to Loop</button>'));
                }
            }
    
            if (!viewOnly) {
                jsPlumb.draggable(div, {
                    containment: '#canvas',
                    grid: [10, 10],
                    drag: function() {
                        jsPlumb.repaintEverything();
                    },
                    stop: function() {
                        updateWorkflow();
                    }
                });
            }
    
            if (div.find('.endpoint').length === 0) {
                console.log('Adding endpoints for node:', node.id);
                jsPlumb.addEndpoint(node.id, {
                    anchor: 'Right',
                    isSource: !viewOnly,
                    isTarget: false,
                    connector: ['Flowchart', { cornerRadius: 15, stub: [20, 30] }], // Sử dụng Flowchart thay vì Bezier
                    maxConnections: 1,
                    endpoint: 'Rectangle', // Đổi sang hình chữ nhật nhỏ
                    paintStyle: { fill: '#34495e', width: 8, height: 8 }, // Màu xám đậm
                    connectorStyle: { 
                        stroke: '#34495e', // Màu xám đậm
                        strokeWidth: 2, // Giữ độ dày
                        dropShadow: { dx: 1, dy: 1, blur: 2, color: 'rgba(0,0,0,0.3)' } // Thêm bóng đổ
                    },
                    connectorOverlays: [
                        ['Label', { 
                            label: 'Flow', 
                            location: 0.5, 
                            cssClass: 'jtk-overlay' 
                        }]
                    ]
                });
    
                jsPlumb.addEndpoint(node.id, {
                    anchor: 'Left',
                    isSource: false,
                    isTarget: !viewOnly,
                    maxConnections: 1,
                    endpoint: 'Rectangle',
                    paintStyle: { fill: '#34495e', width: 8, height: 8 }
                });
            }
    
            if (!viewOnly) {
                div.off('dblclick').on('dblclick', function() {
                    configureNode(node.id);
                });
            } else {
                div.off('dblclick').on('dblclick', function() {
                    const config = JSON.parse($(this).attr('data-config') || '{}');
                    $('#modal-title').text('Cấu hình ' + node.type.replace(/_/g, ' ').toUpperCase());
                    $('#config-fields').empty();
                    for (const [key, value] of Object.entries(config)) {
                        $('#config-fields').append(`
                            <div class="form-group">
                                <label>${key.replace(/_/g, ' ').toUpperCase()}:</label>
                                <p>${value}</p>
                            </div>
                        `);
                    }
                    $('#config-modal').addClass('show animate__fadeInDown');
                    $('#save-config').hide();
                    $('#cancel-config').text('Đóng');
                });
            }
        });
    
        // Tái tạo các kết nối
        workflow.connections.forEach(function(conn) {
            console.log('Rendering connection:', conn);
            try {
                const sourceElement = $('#' + conn.source);
                const targetElement = $('#' + conn.target);
                if (sourceElement.length > 0 && targetElement.length > 0) {
                    const existingConnections = jsPlumb.getConnections({
                        source: conn.source,
                        target: conn.target
                    });
                    if (existingConnections.length === 0) {
                        jsPlumb.connect({
                            source: conn.source,
                            target: conn.target,
                            connector: ['StateMachine'],
                            endpoint: 'Blank',
                            paintStyle: { fill: '#2985E0FF', width: 1, height: 1 },
                            connectorStyle: { 
                                stroke: '#34495e',
                                strokeWidth: 1,
                                dropShadow: { dx: 1, dy: 1, blur: 2, color: 'rgba(0,0,0,0.3)' }
                            },
                            connectorOverlays: [
                                ['Label', { 
                                    label: 'Flow', 
                                    location: 0.5, 
                                    cssClass: 'jtk-overlay' 
                                }]
                            ]
                        });
                    }
                } else {
                    console.warn('Source or target not found in DOM:', conn);
                    if (sourceElement.length === 0) {
                        console.error(`Source node ${conn.source} not found in DOM.`);
                    }
                    if (targetElement.length === 0) {
                        console.error(`Target node ${conn.target} not found in DOM.`);
                    }
                }
            } catch (e) {
                console.error(`Error connecting ${conn.source} to ${conn.target}:`, e);
            }
        });
    
        if (!viewOnly) {
            updateWorkflow();
        }
    
        updateTransform();
    };

    $('#workflow-form').on('submit', function(e) {
        e.preventDefault();
    
        updateWorkflow();
        const workflow = JSON.parse($('#flow').val());
    
        const workflowNameInput = $('#name');
        if (!workflowNameInput.length) {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: 'Không tìm thấy trường nhập tên workflow (id="workflow-name"). Vui lòng kiểm tra HTML!'
            });
            $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
            return;
        }
    
        const workflowName = workflowNameInput.val();
        console.log('Workflow Name (raw):', workflowName);
        if (!workflowName || workflowName.trim() === '' || workflowName === 'undefined') {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: 'Vui lòng nhập tên workflow hợp lệ! Tên không được để trống hoặc chỉ chứa khoảng trắng.'
            });
            $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
            return;
        }
    
        const configErrors = validateAllNodes(workflow.nodes);
        if (configErrors) {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: configErrors
            });
            $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
            return;
        }
    
        const validationError = validateFlowStructure(workflow.nodes, workflow.connections);
        if (validationError) {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: validationError
            });
            $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
            return;
        }
    
        const formData = new FormData();
        formData.append('flow', JSON.stringify(workflow));
        formData.append('name', workflowName.trim());
    
        Object.keys(nodeFiles).forEach(nodeId => {
            const file = nodeFiles[nodeId];
            if (Array.isArray(file)) {
                file.forEach((f, index) => {
                    formData.append(`node_${nodeId}_file_${index}`, f);
                });
            } else {
                formData.append(`node_${nodeId}_file`, file);
            }
        });
    
        for (let pair of formData.entries()) {
            console.log(pair[0] + ': ', pair[1]);
        }
    
        $.ajax({
            url: $(this).attr('action'),
            type: $(this).attr('method'),
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                console.log('Success response:', response);
    
                if (response.messages && response.messages.length > 0) {
                    let messageIndex = 0;
                    const showNextMessage = () => {
                        if (messageIndex < response.messages.length) {
                            Swal.fire({
                                icon: response.success ? 'success' : 'error',
                                title: response.success ? 'Thành công' : 'Lỗi',
                                text: response.messages[messageIndex],
                                confirmButtonText: 'OK'
                            }).then(() => {
                                messageIndex++;
                                showNextMessage();
                            });
                        } else if (response.redirect_url) {
                            window.location.href = response.redirect_url;
                        }
                    };
                    showNextMessage();
                } else if (response.redirect_url) {
                    window.location.href = response.redirect_url;
                } else {
                    $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
                }
            },
            error: function(xhr, status, error) {
                console.log('Error response:', xhr.responseText);
                console.log('Status code:', xhr.status);
                let errorMessages = ['Đã có lỗi xảy ra khi tạo workflow. Vui lòng thử lại.'];
                let redirectUrl = null;
                if (xhr.responseJSON) {
                    if (xhr.responseJSON.messages) {
                        errorMessages = xhr.responseJSON.messages;
                    }
                    if (xhr.responseJSON.redirect_url) {
                        redirectUrl = xhr.responseJSON.redirect_url;
                    }
                }
                let messageIndex = 0;
                const showNextError = () => {
                    if (messageIndex < errorMessages.length) {
                        Swal.fire({
                            icon: 'error',
                            title: 'Lỗi',
                            text: errorMessages[messageIndex],
                            confirmButtonText: 'OK'
                        }).then(() => {
                            messageIndex++;
                            showNextError();
                        });
                    } else if (redirectUrl) {
                        window.location.href = redirectUrl;
                    } else {
                        $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
                    }
                };
                showNextError();
            },
            beforeSend: function() {
                $('#submit-btn').prop('disabled', true).text('Đang xử lý...');
            },
            complete: function() {
                $('#submit-btn').prop('disabled', false).text('Tạo Workflow');
            }
        });
    });

    $('#canvas').empty();
});